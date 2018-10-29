import sys
import time
from collections import defaultdict

import concurrent.futures
import os
from enum import Enum
from functools import partial
from glob import glob
from semstr.convert import FROM_FORMAT, TO_FORMAT, from_text
from semstr.evaluate import EVALUATORS, Scores
from semstr.util.amr import LABEL_ATTRIB, WIKIFIER
from tqdm import tqdm
from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize

from tupa.__version__ import GIT_VERSION
from tupa.config import Config, Iterations
from tupa.model import Model, NODE_LABEL_KEY, ClassifierProperty
from tupa.oracle import Oracle
from tupa.states.state import State
from tupa.traceutil import set_traceback_listener


class ParserException(Exception):
    pass


class ParseMode(Enum):
    train = 1
    dev = 2
    test = 3


class AbstractParser:
    def __init__(self, config, models, training=False, evaluation=False):
        self.config = config
        self.models = models
        self.training = training
        self.evaluation = evaluation
        self.action_count = self.correct_action_count = self.label_count = self.correct_label_count = \
            self.num_tokens = self.f1 = 0
        self.started = time.time()

    @property
    def model(self):
        return self.models[0]

    @model.setter
    def model(self, model):
        self.models[0] = model

    @property
    def duration(self):
        return (time.time() - self.started) or 1.0

    def tokens_per_second(self):
        return self.num_tokens / self.duration


class PassageParser(AbstractParser):
    """ Parser for a single passage, has a state and optionally an oracle """
    def __init__(self, passage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passage = self.out = passage
        self.format = self.passage.extra.get("format") if self.training or self.evaluation else \
            sorted(set.intersection(*map(set, filter(None, (self.model.formats, self.config.args.formats)))) or
                   self.model.formats)[0]
        self.in_format = self.format or "ucca"
        self.out_format = "ucca" if self.format in (None, "text") else self.format
        self.lang = self.passage.attrib.get("lang", self.config.args.lang)
        # Used in verify_passage to optionally ignore a mismatch in linkage nodes:
        self.ignore_node = None if self.config.args.linkage else lambda n: n.tag == layer1.NodeTags.Linkage
        self.state_hash_history = set()
        self.state = self.oracle = self.eval_type = None

    def init(self):
        self.config.set_format(self.in_format)
        WIKIFIER.enabled = self.config.args.wikification
        self.state = State(self.passage)
        # Passage is considered labeled if there are any edges or node labels in it
        edges, node_labels = map(any, zip(*[(n.outgoing, n.attrib.get(LABEL_ATTRIB))
                                            for n in self.passage.layer(layer1.LAYER_ID).all]))
        self.oracle = Oracle(self.passage) if self.training or self.config.args.verify or (
                (self.config.args.verbose > 1 or self.config.args.use_gold_node_labels or self.config.args.action_stats)
                and (edges or node_labels)) else None
        for model in self.models:
            model.init_model(self.config.format, lang=self.lang if self.config.args.multilingual else None)
            if ClassifierProperty.require_init_features in model.classifier_properties:
                model.init_features(self.state, self.training)

    def parse(self, display=True, write=False):
        self.init()
        passage_id = self.passage.ID
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.parse_internal).result(self.config.args.timeout)
            status = "(%d tokens/s)" % self.tokens_per_second()
        except ParserException as e:
            if self.training:
                raise
            self.config.log("%s %s: %s" % (self.config.passage_word, passage_id, e))
            status = "(failed)"
        except concurrent.futures.TimeoutError:
            self.config.log("%s %s: timeout (%fs)" % (self.config.passage_word, passage_id, self.config.args.timeout))
            status = "(timeout)"
        return self.finish(status, display=display, write=write)

    def parse_internal(self):
        """
        Internal method to parse a single passage.
        If training, use oracle to train on given passages. Otherwise just parse with classifier.
        """
        self.config.print("  initial state: %s" % self.state)
        while True:
            if self.config.args.check_loops:
                self.check_loop()
            self.label_node()  # In case root node needs labeling
            true_actions = self.get_true_actions()
            action, predicted_action = self.choose(true_actions)
            self.state.transition(action)
            need_label, label, predicted_label, true_label = self.label_node(action)
            if self.config.args.action_stats:
                try:
                    with open(self.config.args.action_stats, "a") as f:
                        print(",".join(map(str, [predicted_action, action] + list(true_actions.values()))), file=f)
                except OSError:
                    pass
            self.config.print(lambda: "\n".join(["  predicted: %-15s true: %-15s taken: %-15s %s" % (
                predicted_action, "|".join(map(str, true_actions.values())), action, self.state) if self.oracle else
                                          "  action: %-15s %s" % (action, self.state)] + (
                ["  predicted label: %-9s true label: %s" % (predicted_label, true_label) if self.oracle and not
                 self.config.args.use_gold_node_labels else "  label: %s" % label] if need_label else []) + [
                "    " + l for l in self.state.log]))
            if self.state.finished:
                return  # action is Finish (or early update is triggered)

    def get_true_actions(self):
        true_actions = {}
        if self.oracle:
            try:
                true_actions = self.oracle.get_actions(self.state, self.model.actions, create=self.training)
            except (AttributeError, AssertionError) as e:
                if self.training:
                    raise ParserException("Error in getting action from oracle during training") from e
        return true_actions

    def get_true_label(self, node):
        try:
            return self.oracle.get_label(self.state, node) if self.oracle else (None, None)
        except AssertionError as e:
            if self.training:
                raise ParserException("Error in getting label from oracle during training") from e
            return None, None

    def label_node(self, action=None):
        true_label = label = predicted_label = None
        need_label = self.state.need_label  # Label action that requires a choice of label
        if need_label:
            true_label, raw_true_label = self.get_true_label(action or need_label)
            label, predicted_label = self.choose(true_label, NODE_LABEL_KEY, "node label")
            self.state.label_node(raw_true_label if label == true_label else label)
        return need_label, label, predicted_label, true_label

    def choose(self, true, axis=None, name="action"):
        if axis is None:
            axis = self.model.axis
        elif axis == NODE_LABEL_KEY and self.config.args.use_gold_node_labels:
            return true, true
        labels = self.model.classifier.labels[axis]
        if axis == NODE_LABEL_KEY:
            true_keys = (labels[true],) if self.oracle else ()  # Must be before score()
            is_valid = self.state.is_valid_label
        else:
            true_keys = None
            is_valid = self.state.is_valid_action
        scores, features = self.model.score(self.state, axis)
        for model in self.models[1:]:  # Ensemble if given more than one model; align label order and add scores
            label_scores = dict(zip(model.classifier.labels[axis].all, self.model.score(self.state, axis)[0]))
            scores += [label_scores.get(a, 0) for a in labels.all]  # Product of Experts, assuming log(softmax)
        self.config.print(lambda: "  %s scores: %s" % (name, tuple(zip(labels.all, scores))), level=4)
        try:
            label = pred = self.predict(scores, labels.all, is_valid)
        except StopIteration as e:
            raise ParserException("No valid %s available\n%s" % (name, self.oracle.log if self.oracle else "")) from e
        label, is_correct, true_keys, true_values = self.correct(axis, label, pred, scores, true, true_keys)
        if self.training:
            if not (is_correct and ClassifierProperty.update_only_on_error in self.model.classifier_properties):
                assert not self.model.is_finalized, "Updating finalized model"
                self.model.classifier.update(
                    features, axis=axis, true=true_keys, pred=labels[pred] if axis == NODE_LABEL_KEY else pred.id,
                    importance=[self.config.args.swap_importance if a.is_swap else 1 for a in true_values] or None)
            if not is_correct and self.config.args.early_update:
                self.state.finished = True
        for model in self.models:
            model.classifier.finished_step(self.training)
            if axis != NODE_LABEL_KEY:
                model.classifier.transition(label, axis=axis)
        return label, pred

    def correct(self, axis, label, pred, scores, true, true_keys):
        true_values = is_correct = ()
        if axis == NODE_LABEL_KEY:
            if self.oracle:
                is_correct = (label == true)
                if is_correct:
                    self.correct_label_count += 1
                else:
                    label = true
            self.label_count += 1
        else:  # action
            true_keys, true_values = map(list, zip(*true.items())) if true else (None, None)
            label = true.get(pred.id)
            is_correct = (label is not None)
            if is_correct:
                self.correct_action_count += 1
            else:
                label = true_values[scores[true_keys].argmax()] if self.training else pred
            self.action_count += 1
        return label, is_correct, true_keys, true_values

    @staticmethod
    def predict(scores, values, is_valid=None):
        """
        Choose action/label based on classifier
        Usually the best action/label is valid, so max is enough to choose it in O(n) time
        Otherwise, sorts all the other scores to choose the best valid one in O(n lg n)
        :return: valid action/label with maximum probability according to classifier
        """
        return next(filter(is_valid, (values[i] for i in PassageParser.generate_descending(scores))))

    @staticmethod
    def generate_descending(scores):
        yield scores.argmax()
        yield from scores.argsort()[::-1]  # Contains the max, but otherwise items might be missed (different order)

    def finish(self, status, display=True, write=False):
        self.model.classifier.finished_item(self.training)
        for model in self.models[1:]:
            model.classifier.finished_item(renew=False)  # So that dynet.renew_cg happens only once
        if not self.training or self.config.args.verify:
            self.out = self.state.create_passage(verify=self.config.args.verify, format=self.out_format)
        if write:
            for out_format in self.config.args.formats or [self.out_format]:
                if self.config.args.normalize and out_format == "ucca":
                    normalize(self.out)
                ioutil.write_passage(self.out, output_format=out_format, binary=out_format == "pickle",
                                     outdir=self.config.args.outdir, prefix=self.config.args.prefix,
                                     converter=get_output_converter(out_format), verbose=self.config.args.verbose,
                                     append=self.config.args.join, basename=self.config.args.join)
        if self.oracle and self.config.args.verify:
            self.verify(self.out, self.passage)
        ret = (self.out,)
        if self.evaluation:
            ret += (self.evaluate(self.evaluation),)
            status = "%-14s %s F1=%.3f" % (status, self.eval_type, self.f1)
        if display:
            self.config.print("%s%.3fs %s" % (self.accuracy_str, self.duration, status), level=1)
        return ret

    @property
    def accuracy_str(self):
        if self.oracle and self.action_count:
            accuracy_str = "a=%-14s" % percents_str(self.correct_action_count, self.action_count)
            if self.label_count:
                accuracy_str += " l=%-14s" % percents_str(self.correct_label_count, self.label_count)
            return "%-33s" % accuracy_str
        return ""

    def evaluate(self, mode=ParseMode.test):
        if self.format:
            self.config.print("Converting to %s and evaluating..." % self.format)
        self.eval_type = UNLABELED if self.config.is_unlabeled(self.in_format) else LABELED
        evaluator = EVALUATORS.get(self.format, evaluate_ucca)
        score = evaluator(self.out, self.passage, converter=get_output_converter(self.format),
                          verbose=self.out and self.config.args.verbose > 3,
                          constructions=self.config.args.constructions,
                          eval_types=(self.eval_type,) if mode is ParseMode.dev else (LABELED, UNLABELED))
        self.f1 = average_f1(score, self.eval_type)
        score.lang = self.lang
        return score

    def check_loop(self):
        """
        Check if the current state has already occurred, indicating a loop
        """
        h = hash(self.state)
        assert h not in self.state_hash_history, \
            "\n".join(["Transition loop", self.state.str("\n")] + [self.oracle.str("\n")] if self.oracle else ())
        self.state_hash_history.add(h)

    def verify(self, guessed, ref):
        """
        Compare predicted passage to true passage and raise an exception if they differ
        :param ref: true passage
        :param guessed: predicted passage to compare
        """
        assert ref.equals(guessed, ignore_node=self.ignore_node), \
            "Failed to produce true passage" + (diffutil.diff_passages(ref, guessed) if self.training else "")

    @property
    def num_tokens(self):
        return len(set(self.state.terminals).difference(self.state.buffer))  # To count even incomplete parses

    @num_tokens.setter
    def num_tokens(self, _):
        pass


class BatchParser(AbstractParser):
    """ Parser for a single training iteration or single pass over dev/test passages """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_per_format = defaultdict(int)
        self.num_passages = 0

    def parse(self, passages, display=True, write=False):
        passages, total = generate_and_len(single_to_iter(passages))
        if self.config.args.ignore_case:
            passages = to_lower_case(passages)
        pr_width = len(str(total))
        id_width = 1
        passages = self.add_progress_bar(textutil.annotate_all(
            passages, as_array=True, lang=self.config.args.lang, verbose=self.config.args.verbose > 2,
            vocab=self.model.config.vocab(lang=self.config.args.lang)), display=display)
        for i, passage in enumerate(passages, start=1):
            parser = PassageParser(passage, self.config, self.models, self.training, self.evaluation)
            if self.config.args.verbose and display:
                progress = "%3d%% %*d/%d" % (i / total * 100, pr_width, i, total) if total and i <= total else "%d" % i
                id_width = max(id_width, len(str(passage.ID)))
                print("%s %2s %-6s %-*s" % (progress, parser.lang, parser.in_format, id_width, passage.ID),
                      end=self.config.line_end)
            else:
                passages.set_description()
                postfix = {parser.lang + " " + parser.in_format: passage.ID}
                if display:
                    postfix["|t/s|"] = self.tokens_per_second()
                    if self.correct_action_count:
                        postfix["|a|"] = percents_str(self.correct_action_count, self.action_count, fraction=False)
                    if self.correct_label_count:
                        postfix["|l|"] = percents_str(self.correct_label_count, self.label_count, fraction=False)
                    if self.evaluation and self.num_passages:
                        postfix["|F1|"] = self.f1 / self.num_passages
                passages.set_postfix(**postfix)
            self.seen_per_format[parser.in_format] += 1
            if self.training and self.config.args.max_training_per_format and \
                    self.seen_per_format[parser.in_format] > self.config.args.max_training_per_format:
                self.config.print("skipped", level=1)
                continue
            assert not (self.training and parser.in_format == "text"), "Cannot train on unannotated plain text"
            yield parser.parse(display=display, write=write)
            self.update_counts(parser)
        if self.num_passages and display:
            self.summary()

    def add_progress_bar(self, it, total=None, display=True):
        return it if self.config.args.verbose and display else tqdm(
            it, unit=self.config.passages_word, total=total, file=sys.stdout, desc="Initializing")

    def update_counts(self, parser):
        self.correct_action_count += parser.correct_action_count
        self.action_count += parser.action_count
        self.correct_label_count += parser.correct_label_count
        self.label_count += parser.label_count
        self.num_tokens += parser.num_tokens
        self.num_passages += 1
        self.f1 += parser.f1

    def summary(self):
        print("Parsed %d%s" % (self.num_passages, self.config.passages_word))
        if self.correct_action_count:
            accuracy_str = percents_str(self.correct_action_count, self.action_count, "correct actions ")
            if self.label_count:
                accuracy_str += ", " + percents_str(self.correct_label_count, self.label_count, "correct labels ")
            print("Overall %s" % accuracy_str)
        print("Total time: %.3fs (average time/%s: %.3fs, average tokens/s: %d)" % (
            self.duration, self.config.passage_word, self.time_per_passage(),
            self.tokens_per_second()), flush=True)

    def time_per_passage(self):
        return self.duration / self.num_passages


class Parser(AbstractParser):
    """ Main class to implement transition-based UCCA parser """
    def __init__(self, model_files=(), config=None, beam=1):
        super().__init__(config=config or Config(),
                         models=list(map(Model, (model_files,) if isinstance(model_files, str) else
                                         model_files or (config.args.classifier,))))
        self.beam = beam  # Currently unused
        self.best_score = self.dev = self.test = self.iteration = self.epoch = self.batch = None
        self.trained = self.save_init = False

    def train(self, passages=None, dev=None, test=None, iterations=1):
        """
        Train parser on given passages
        :param passages: iterable of passages to train on
        :param dev: iterable of passages to tune on
        :param test: iterable of passages that would be tested on after train finished
        :param iterations: iterable of Iterations objects whose i attributes are the number of iterations to perform
        """
        self.trained = True
        self.dev = dev
        self.test = test
        if passages:
            self.init_train()
            iterations = [i if isinstance(i, Iterations) else Iterations(i)
                          for i in (iterations if hasattr(iterations, "__iter__") else (iterations,))]
            if any(i.epochs >= j.epochs for i, j in zip(iterations[:-1], iterations[1:])):
                raise ValueError("Arguments to --iterations must be increasing: " + " ".join(map(str, iterations)))
            self.config.args.iterations = iterations
            end = None
            for self.iteration, it in enumerate(iterations, start=1):
                start = self.model.classifier.epoch + 1 if self.model.classifier else 1
                if end and start < end + 1:
                    print("Dropped %d epochs because best score was on %d" % (end - start + 1, start - 1))
                end = it.epochs + 1
                self.config.update_iteration(it)
                if end < start + 1:
                    print("Skipping %s, already trained %s epochs" % (it, start - 1))
                    continue
                for self.epoch in range(start, end):
                    print("Training epoch %d of %d: " % (self.epoch, end - 1))
                    self.config.random.shuffle(passages)
                    if not sum(1 for _ in self.parse(passages, mode=ParseMode.train)):
                        raise ParserException("Could not train on any passage")
                    yield self.eval_and_save(self.iteration == len(iterations) and self.epoch == end - 1,
                                             finished_epoch=True)
                print("Trained %d epochs" % (end - 1))
                if dev:
                    if self.iteration < len(iterations):
                        if self.model.is_retrainable:
                            self.model.load(is_finalized=False)  # Load best model to prepare for next iteration
                    elif test:
                        self.model.load()  # Load best model to prepare for test
        else:  # No passages to train on, just load model
            for model in self.models:
                model.load()
            self.print_config()

    def init_train(self):
        assert len(self.models) == 1, "Can only train one model at a time"
        if self.model.is_retrainable:
            try:
                self.model.load(is_finalized=False)
            except FileNotFoundError:
                print("not found, starting from untrained model.")
        self.print_config()
        self.best_score = self.model.classifier.best_score if self.model.classifier else 0

    def eval_and_save(self, last=False, finished_epoch=False):
        scores = None
        model = self.model
        # noinspection PyAttributeOutsideInit
        self.model = finalized = model.finalize(finished_epoch=finished_epoch)
        if self.dev:
            if not self.best_score:
                self.save(finalized)
            average_score, scores = self.eval(self.dev, ParseMode.dev, self.config.args.devscores)
            if average_score >= self.best_score:
                print("Better than previous best score (%.3f)" % self.best_score)
                finalized.classifier.best_score = average_score
                if self.best_score:
                    self.save(finalized)
                self.best_score = average_score
                if self.test and self.test is not True:  # There are actual passages to parse
                    self.eval(self.test, ParseMode.test, self.config.args.testscores, display=False)
            else:
                print("Not better than previous best score (%.3f)" % self.best_score)
        elif last or self.config.args.save_every is not None:
            self.save(finalized)
        if not last:
            finalized.restore(model)  # Restore non-finalized model
        return scores

    def save(self, model):
        self.config.save(model.filename)
        model.save(save_init=self.save_init)

    def eval(self, passages, mode, scores_filename, display=True):
        print("Evaluating on %s passages" % mode.name)
        passage_scores = [s for _, s in self.parse(passages, mode=mode, evaluate=True, display=display)]
        scores = Scores(passage_scores)
        average_score = average_f1(scores)
        prefix = ".".join(map(str, [self.iteration, self.epoch] + (
            [self.batch] if self.config.args.save_every else [])))
        if display:
            print("Evaluation %s, average %s F1 score on %s: %.3f%s" % (prefix, get_eval_type(scores), mode.name,
                                                                        average_score, scores.details(average_f1)))
        print_scores(scores, scores_filename, prefix=prefix, prefix_title="iteration")
        return average_score, scores

    def parse(self, passages, mode=ParseMode.test, evaluate=False, display=True, write=False):
        """
        Parse given passages
        :param passages: iterable of passages to parse
        :param mode: ParseMode value.
                     If train, use oracle to train on given passages.
                     Otherwise, just parse with classifier.
        :param evaluate: whether to evaluate parsed passages with respect to given ones.
                           Only possible when given passages are annotated.
        :param display: whether to display information on each parsed passage
        :param write: whether to write output passages to file
        :return: generator of parsed passages (or in train mode, the original ones),
                 or, if evaluation=True, of pairs of (Passage, Scores).
        """
        self.batch = 0
        assert mode in ParseMode, "Invalid parse mode: %s" % mode
        training = (mode is ParseMode.train)
        if not training and not self.trained:
            yield from self.train()  # Try to load model from file
        parser = BatchParser(self.config, self.models, training, mode if mode is ParseMode.dev else evaluate)
        for i, passage in enumerate(parser.parse(passages, display=display, write=write), start=1):
            if training and self.config.args.save_every and i % self.config.args.save_every == 0:
                self.eval_and_save()
                self.batch += 1
            yield passage

    def print_config(self):
        self.config.print("tupa %s" % (self.model.config if self.model else self.config), level=0)


def train_test(train_passages, dev_passages, test_passages, args, model_suffix=""):
    """
    Train and test parser on given passage
    :param train_passages: passage to train on
    :param dev_passages: passages to evaluate on every iteration
    :param test_passages: passages to test on after training
    :param args: extra argument
    :param model_suffix: string to append to model filename before file extension
    :return: generator of Scores objects: dev scores for each training iteration (if given dev), and finally test scores
    """
    model_files = [base + model_suffix + ext for base, ext in map(os.path.splitext, args.models or (args.classifier,))]
    p = Parser(model_files=model_files, config=Config(), beam=args.beam)
    yield from filter(None, p.train(train_passages, dev=dev_passages, test=test_passages, iterations=args.iterations))
    if test_passages:
        if args.train or args.folds:
            print("Evaluating on test passages")
        passage_scores = []
        evaluate = args.evaluate or train_passages
        for result in p.parse(test_passages, evaluate=evaluate, write=args.write):
            _, *score = result
            passage_scores += score
        if passage_scores:
            scores = Scores(passage_scores)
            if args.verbose <= 1 or len(passage_scores) > 1:
                print("\nAverage %s F1 score on test: %.3f" % (get_eval_type(scores), average_f1(scores)))
                print("Aggregated scores:")
                scores.print()
            print_scores(scores, args.testscores)
            yield scores


def get_output_converter(out_format, default=None):
    converter = TO_FORMAT.get(out_format)
    return partial(converter, wikification=Config().args.wikification,
                   verbose=Config().args.verbose > 2) if converter else default


def percents_str(part, total, infix="", fraction=True):
    ret = "%d%%" % (100 * part / total)
    if fraction:
        ret += " %s(%d/%d)" % (infix, part, total)
    return ret


def print_scores(scores, filename, prefix=None, prefix_title=None):
    if filename:
        print_title = not os.path.exists(filename)
        try:
            with open(filename, "a") as f:
                if print_title:
                    titles = scores.titles()
                    if prefix_title is not None:
                        titles = [prefix_title] + titles
                    print(",".join(titles), file=f)
                fields = scores.fields()
                if prefix is not None:
                    fields.insert(0, prefix)
                print(",".join(fields), file=f)
        except OSError:
            pass


def single_to_iter(it):
    return it if hasattr(it, "__iter__") else (it,)  # Single passage given


def generate_and_len(it):
    return it, (len(it) if hasattr(it, "__len__") else None)


def to_lower_case(passages):
    for passage in passages:
        for terminal in passage.layer(layer0.LAYER_ID).all:
            terminal.text = terminal.text.lower()
        yield passage


def average_f1(scores, eval_type=None):
    for e in (eval_type or get_eval_type(scores),) + EVAL_TYPES:
        try:
            return scores.average_f1(e)
        except ValueError:
            pass
    return 0


def get_eval_type(scores):
    return UNLABELED if Config().is_unlabeled(scores.format) else LABELED


# Marks input passages as text so that we don't accidentally train on them
def from_text_format(*args, **kwargs):
    for passage in from_text(*args, **kwargs):
        passage.extra["format"] = "text"
        yield passage


CONVERTERS = {k: partial(c, annotate=True) for k, c in FROM_FORMAT.items()}
CONVERTERS[""] = CONVERTERS["txt"] = from_text_format


def read_passages(args, files):
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, sentences=args.sentences, paragraphs=args.paragraphs,
                                      converters=CONVERTERS, lang=Config().args.lang)


# noinspection PyTypeChecker,PyStringFormat
def main_generator():
    args = Config().args
    assert args.passages or args.train, "Either passages or --train is required (use -h for help)"
    assert args.models or args.train or args.folds, "Either --model or --train or --folds is required"
    assert not (args.train or args.dev) or not args.folds, "--train and --dev are incompatible with --folds"
    assert args.train or not args.dev, "--dev is only possible together with --train"
    if args.folds:
        fold_scores = []
        all_passages = list(read_passages(args, args.passages))
        assert len(all_passages) >= args.folds, \
            "%d folds are not possible with only %d passages" % (args.folds, len(all_passages))
        Config().random.shuffle(all_passages)
        folds = [all_passages[i::args.folds] for i in range(args.folds)]
        for i in range(args.folds):
            print("Fold %d of %d:" % (i + 1, args.folds))
            dev_passages = folds[i]
            test_passages = folds[(i + 1) % args.folds]
            train_passages = [passage for fold in folds if fold is not dev_passages and fold is not test_passages
                              for passage in fold]
            s = list(train_test(train_passages, dev_passages, test_passages, args, "_%d" % i))
            if s and s[-1] is not None:
                fold_scores.append(s[-1])
        if fold_scores:
            scores = Scores(fold_scores)
            print("Average test F1 score for each fold: " + ", ".join("%.3f" % average_f1(s) for s in fold_scores))
            print("Aggregated scores across folds:\n")
            scores.print()
            yield scores
    else:  # Simple train/dev/test by given arguments
        train_passages, dev_passages, test_passages = [read_passages(args, arg) for arg in
                                                       (args.train, args.dev, args.passages)]
        yield from train_test(train_passages, dev_passages, test_passages, args)


def main():
    print("TUPA version " + GIT_VERSION)
    set_traceback_listener()
    list(main_generator())


if __name__ == "__main__":
    main()
