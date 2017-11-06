import concurrent.futures
import os
import sys
import time
from collections import defaultdict
from enum import Enum
from functools import partial
from glob import glob

from tqdm import tqdm
from ucca import diffutil, ioutil, textutil, layer1, evaluation
from ucca.convert import from_text, to_text

from scheme.convert import FROM_FORMAT, TO_FORMAT
from scheme.evaluate import EVALUATORS, Scores
from scheme.util.amr import LABEL_ATTRIB, WIKIFIER
from tupa.config import Config
from tupa.model import Model, NODE_LABEL_KEY, ClassifierProperty
from tupa.oracle import Oracle
from tupa.states.state import State


class ParserException(Exception):
    pass


class ParseMode(Enum):
    train = 1
    dev = 2
    test = 3


class Parser(object):

    """
    Main class to implement transition-based UCCA parser
    """
    def __init__(self, model_file=None, model_type=None, beam=1):
        self.args = Config().args
        self.state = self.oracle = None  # State and Oracle objects created at each parse
        self.action_count = self.correct_action_count = self.total_actions = self.total_correct_actions = 0
        self.label_count = self.correct_label_count = self.total_labels = self.total_correct_labels = 0
        self.model = Model(model_type, model_file)
        self.beam = beam  # Currently unused
        self.state_hash_history = None  # For loop checking
        # Used in verify_passage to optionally ignore a mismatch in linkage nodes:
        self.ignore_node = None if self.args.linkage else lambda n: n.tag == layer1.NodeTags.Linkage
        self.best_score = self.dev = self.iteration = self.eval_index = None
        self.training = self.trained = False

    def train(self, passages=None, dev=None, test=None, iterations=1):
        """
        Train parser on given passages
        :param passages: iterable of passages to train on
        :param dev: iterable of passages to tune on
        :param test: iterable of passages that would be tested on after train finished
        :param iterations: number of iterations to perform
        """
        self.trained = True
        if passages:
            if ClassifierProperty.trainable_after_saving in self.model.get_classifier_properties():
                try:
                    self.model.load(finalized=False)
                except FileNotFoundError:
                    print("not found, starting from untrained model.")
            for f in self.args.formats or ():
                Config().set_format(f)
                self.model.init_model()
            print_config()
            self.best_score = 0
            self.dev = dev
            for self.iteration in range(1, iterations + 1):
                self.eval_index = 0
                print("Training iteration %d of %d: " % (self.iteration, iterations))
                Config().random.shuffle(passages)
                list(self.parse(passages, mode=ParseMode.train))
                yield self.eval_and_save(self.iteration == iterations, finished_epoch=True)
            print("Trained %d iterations" % iterations)
        if dev and test or not passages:
            self.model.load()  # Load best model (on dev) to prepare for test
            if not passages:
                print_config()

    def eval_and_save(self, last=False, finished_epoch=False):
        scores = None
        model = self.model
        self.model = self.model.finalize(finished_epoch=finished_epoch)
        if self.dev:
            if not self.best_score:
                self.model.save()
            print("Evaluating on dev passages")
            passage_scores = [s for _, s in self.parse(self.dev, mode=ParseMode.dev, evaluate=True)]
            scores = Scores(passage_scores)
            average_score = scores.average_f1()
            prefix = ".".join(map(str, [self.iteration] + ([self.eval_index] if self.args.save_every else [])))
            score_details = "" if len(scores.scores_by_format) < 2 else " (" + ", ".join(
                "%.3f" % s.average_f1() for f, s in scores.scores_by_format) + ")"
            print("Evaluation %s, average labeled F1 score on dev: %.3f%s" % (prefix, average_score, score_details))
            print_scores(scores, self.args.devscores, prefix=prefix, prefix_title="iteration")
            if average_score >= self.best_score:
                print("Better than previous best score (%.3f)" % self.best_score)
                if self.best_score:
                    self.model.save()
                self.best_score = average_score
            else:
                print("Not better than previous best score (%.3f)" % self.best_score)
        elif last or self.args.save_every is not None:
            self.model.save()
        if not last:
            self.model.restore(model)  # Restore non-finalized model
        return scores

    def parse(self, passages, mode=ParseMode.test, evaluate=False):
        """
        Parse given passages
        :param passages: iterable of passages to parse
        :param mode: ParseMode value.
                     If train, use oracle to train on given passages.
                     Otherwise, just parse with classifier.
        :param evaluate: whether to evaluate parsed passages with respect to given ones.
                         Only possible when given passages are annotated.
        :return: generator of parsed passages (or in train mode, the original ones),
                 or, if evaluate=True, of pairs of (Passage, Scores).
        """
        assert mode in ParseMode, "Invalid parse mode: %s" % mode
        self.training = (mode is ParseMode.train)
        if not self.training and not self.trained:
            list(self.train())  # Try to load model from file
        passage_word = "sentence" if self.args.sentences else "paragraph" if self.args.paragraphs else "passage"
        self.total_actions = 0
        self.total_correct_actions = 0
        total_duration = 0
        total_tokens = 0
        passage_index = 0
        if not hasattr(passages, "__iter__"):  # Single passage given
            passages = (passages,)
        passages_iter = enumerate(passages, start=1)
        for passage_index, passage in passages_iter if self.args.verbose else tqdm(
                passages_iter, unit=passage_word, total=len(passages) if hasattr(passages, "__len__") else None,
                file=sys.stdout):
            edges, node_labels = map(any, zip(*[(n.outgoing, n.attrib.get(LABEL_ATTRIB))
                                                for n in passage.layer(layer1.LAYER_ID).all]))
            # Passage is considered labeled if there are any edges or node labels in it
            passage_format = passage.extra.get("format") or "ucca"
            if self.args.verbose:
                print("%-6s %s %-7s" % (passage_format, passage_word, passage.ID), end=Config().line_end, flush=True)
            assert not (self.training and passage_format == "text"), "Cannot train on unannotated plain text"
            started = time.time()
            self.action_count = self.correct_action_count = self.label_count = self.correct_label_count = 0
            textutil.annotate(passage, verbose=self.args.verbose > 2)  # tag POS and parse dependencies
            Config().set_format(passage_format)
            WIKIFIER.enabled = self.args.wikification
            self.state = State(passage)
            self.state_hash_history = set()
            self.oracle = Oracle(passage) if self.training or ((self.args.verbose > 1 or self.args.use_gold_node_labels)
                                                               and (edges or node_labels)) or self.args.verify else None
            self.model.init_model()
            if ClassifierProperty.require_init_features in self.model.get_classifier_properties():
                axes = [Config().format]
                if self.args.node_labels and not self.args.use_gold_node_labels:
                    axes.append(NODE_LABEL_KEY)
                self.model.init_features(self.state, axes, self.training)
            status = None
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(self.parse_passage).result(self.args.timeout)  # This does the actual parsing
            except ParserException as e:
                if self.training:
                    raise
                Config().log("%s %s: %s" % (passage_word, passage.ID, e))
                status = "failed"
            except concurrent.futures.TimeoutError:
                Config().log("%s %s: timeout (%fs)" % (passage_word, passage.ID, self.args.timeout))
                status = "timeout"
            guessed = self.state.create_passage(verify=self.args.verify) if not self.training or self.args.verify else \
                passage
            duration = time.time() - started
            total_duration += duration
            num_tokens = len(set(self.state.terminals).difference(self.state.buffer))
            total_tokens += num_tokens
            if self.oracle:  # We have an oracle to verify by
                if status is None and self.args.verify:
                    self.verify_passage(guessed, passage)
                if self.action_count:
                    accuracy_str = percents_str(self.correct_action_count, self.action_count)
                    if self.label_count:
                        accuracy_str += " " + percents_str(self.correct_label_count, self.label_count)
                    if self.args.verbose:
                        print("%-30s" % accuracy_str, end=Config().line_end)
            if self.args.verbose:
                if status is None:
                    status = "%d tokens/s" % (num_tokens / duration)
                print("%0.3fs%-15s%s" % (duration, " (" + status + ")", Config().line_end), end="")
                if self.oracle:
                    print(Config().line_end, flush=True)
            self.model.classifier.finished_item(self.training)
            self.total_correct_actions += self.correct_action_count
            self.total_actions += self.action_count
            self.total_correct_labels += self.correct_label_count
            self.total_labels += self.label_count
            if self.training and self.args.save_every and passage_index % self.args.save_every == 0:
                self.eval_and_save()
                self.eval_index += 1
                self.training = True
            yield (guessed, self.evaluate_passage(guessed, passage)) if evaluate else guessed

        if passages:
            print("Parsed %d %ss" % (passage_index, passage_word))
            if self.oracle and self.total_actions:
                accuracy_str = percents_str(self.total_correct_actions, self.total_actions, "correct actions ")
                if self.total_labels:
                    accuracy_str += ", " + percents_str(self.total_correct_labels, self.total_labels, "correct labels ")
                print("Overall %s on %s" % (accuracy_str, mode.name))
            if total_duration:
                print("Total time: %.3fs (average time/%s: %.3fs, average tokens/s: %d)" % (
                    total_duration, passage_word, total_duration / passage_index,
                    total_tokens / total_duration), flush=True)

    def parse_passage(self):
        """
        Internal method to parse a single passage.
        If training, use oracle to train on given passages. Otherwise just parse with classifier.
        """
        if self.args.verbose > 2:
            print("  initial state: %s" % self.state)
        while True:
            if self.args.check_loops:
                self.check_loop()
            features = self.model.feature_extractor.extract_features(self.state)
            true_actions = self.get_true_actions()
            action, predicted_action = self.choose_action(features, true_actions)
            try:
                self.state.transition(action)
            except AssertionError as e:
                raise ParserException("Invalid transition: %s %s" % (action, self.state)) from e
            true_label = label = predicted_label = None
            if self.state.need_label:  # Label action that requires a choice of label
                true_label, raw_true_label = self.get_true_label(action)
                label, predicted_label = self.choose_label(features, true_label)
                self.state.label_node(raw_true_label if label == true_label else label)
            self.model.classifier.finished_step(self.training)
            if self.args.verbose > 2:
                if self.oracle:
                    print("  predicted: %-15s true: %-15s taken: %-15s %s" % (
                        predicted_action, "|".join(map(str, true_actions.values())), action, self.state))
                else:
                    print("  action: %-15s %s" % (action, self.state))
                if true_label or label or predicted_label:
                    if self.oracle and not self.args.use_gold_node_labels:
                        print("  predicted label: %-15s true label: %-15s" % (predicted_label, true_label))
                    else:
                        print("  label: %-15s" % label)
                for line in self.state.log:
                    print("    " + line)
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

    def choose_action(self, features, true_actions):
        scores = self.model.classifier.score(features, axis=Config().format)  # Returns NumPy array
        if self.args.verbose > 3:
            print("  action scores: " + ",".join(("%s: %g" % x for x in zip(self.model.actions.all, scores))))
        try:
            predicted_action = self.predict(scores, self.model.actions.all, self.state.is_valid_action, unit="action")
        except StopIteration as e:
            raise ParserException("No valid action available\n%s" % (self.oracle.log if self.oracle else "")) from e
        action = true_actions.get(predicted_action.id)
        is_correct = (action is not None)
        if is_correct:
            self.correct_action_count += 1
        else:
            action = Config().random.choice(list(true_actions.values())) if self.training else predicted_action
        if self.training and not (
                    is_correct and ClassifierProperty.update_only_on_error in self.model.get_classifier_properties()):
            best_action = self.predict(scores[list(true_actions.keys())], list(true_actions.values()))
            self.model.classifier.update(features, axis=Config().format, pred=predicted_action.id, true=best_action.id,
                                         importance=self.args.swap_importance if best_action.is_swap else 1)
        if self.training and not is_correct and self.args.early_update:
            self.state.finished = True
        self.action_count += 1
        return action, predicted_action

    def get_true_label(self, action):
        try:
            return self.oracle.get_label(self.state, action) if self.oracle else (None, None)
        except AssertionError as e:
            if self.training:
                raise ParserException("Error in getting label from oracle during training") from e

    def choose_label(self, features, true_label):
        true_id = self.model.labels[true_label] if self.oracle else None  # Needs to happen before score()
        if self.args.use_gold_node_labels:
            return true_label, true_label
        scores = self.model.classifier.score(features, axis=NODE_LABEL_KEY)
        if self.args.verbose > 3:
            print("  label scores: " + ",".join(("%s: %g" % x for x in zip(self.model.labels.all, scores))))
        label = predicted_label = self.predict(scores, self.model.labels.all, self.state.is_valid_label, unit="label")
        if self.oracle:
            is_correct = (label == true_label)
            if is_correct:
                self.correct_label_count += 1
            if self.training and not (is_correct and ClassifierProperty.update_only_on_error in
                                      self.model.get_classifier_properties()):
                self.model.classifier.update(features, axis=NODE_LABEL_KEY, pred=self.model.labels[label], true=true_id)
                label = true_label
        self.label_count += 1
        return label, predicted_label

    def predict(self, scores, values, is_valid=None, unit=None):
        """
        Choose action/label based on classifier
        Usually the best action/label is valid, so max is enough to choose it in O(n) time
        Otherwise, sorts all the other scores to choose the best valid one in O(n lg n)
        :return: valid action/label with maximum probability according to classifier
        """
        indices = (values[i] for i in self.generate_descending(scores))
        if unit and self.args.verbose > 2:
            print("Finding valid %s..." % unit)
            indices = tqdm(indices, total=len(scores), unit=" " + unit + "s", file=sys.stdout)
        return next(filter(is_valid, indices))

    @staticmethod
    def generate_descending(scores):
        yield scores.argmax()
        yield from scores.argsort()[::-1]  # Contains the max, but otherwise items might be missed (different order)

    def check_loop(self):
        """
        Check if the current state has already occurred, indicating a loop
        """
        h = hash(self.state)
        assert h not in self.state_hash_history, \
            "\n".join(["Transition loop", self.state.str("\n")] + [self.oracle.str("\n")] if self.oracle else ())
        self.state_hash_history.add(h)

    def evaluate_passage(self, guessed, ref):
        ref_format = ref.extra.get("format")
        if self.args.verbose > 2 and ref_format:
            print("Converting to %s and evaluating..." % ref_format, flush=True)
        score = EVALUATORS.get(ref_format, evaluation).evaluate(
            guessed, ref, converter=get_output_converter(ref_format),
            verbose=guessed and self.args.verbose > 3, constructions=self.args.constructions)
        if self.args.verbose:
            print("F1=%.3f" % score.average_f1(), flush=True)
        return score

    def verify_passage(self, guessed, ref):
        """
        Compare predicted passage to true passage and raise an exception if they differ
        :param ref: true passage
        :param guessed: predicted passage to compare
        """
        assert ref.equals(guessed, ignore_node=self.ignore_node), \
            "Failed to produce true passage" + (diffutil.diff_passages(ref, guessed) if self.training else "")


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
    model_base, model_ext = os.path.splitext(args.model or args.classifier)
    p = Parser(model_file=model_base + model_suffix + model_ext, model_type=args.classifier, beam=args.beam)
    yield from filter(None, p.train(train_passages, dev=dev_passages, test=test_passages, iterations=args.iterations))
    if test_passages:
        if args.train or args.folds:
            print("Evaluating on test passages")
        passage_scores = []
        evaluate = args.evaluate or train_passages
        for result in p.parse(test_passages, evaluate=evaluate):
            if evaluate:
                guessed_passage, score = result
                passage_scores.append(score)
            else:
                guessed_passage = result
                if args.verbose:
                    print()
            if guessed_passage is not None and args.write:
                passage_format = guessed_passage.extra.get("format")
                for out_format in args.formats or ("ucca",) if passage_format in (None, "text") else (passage_format,):
                    ioutil.write_passage(guessed_passage, output_format=out_format, binary=out_format == "pickle",
                                         outdir=args.outdir, prefix=args.prefix,
                                         converter=get_output_converter(out_format, default=to_text))
        if passage_scores:
            scores = Scores(passage_scores)
            if args.verbose <= 1 or len(passage_scores) > 1:
                print("\nAverage labeled F1 score on test: %.3f" % scores.average_f1())
                print("Aggregated scores:")
                scores.print()
            print_scores(scores, args.testscores)
            yield scores


def get_output_converter(out_format, default=None):
    converter = TO_FORMAT.get(out_format)
    return partial(converter, wikification=Config().args.wikification,
                   verbose=Config().args.verbose > 2) if converter else default


def percents_str(part, total, infix=""):
    return "%d%% %s(%d/%d)" % (100 * part / total, infix, part, total)


def print_scores(scores, filename, prefix=None, prefix_title=None):
    if filename:
        print_title = not os.path.exists(filename)
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


def print_config():
    print("%s %s" % (os.path.basename(__file__), Config()))


class TextReader:  # Marks input passages as text so that we don't accidentally train on them
    def __call__(self, *args, **kwargs):
        for passage in from_text(*args, **kwargs):
            passage.extra["format"] = "text"
            yield passage


def read_passages(args, files):
    expanded = [f for pattern in files for f in glob(pattern) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, args.sentences, args.paragraphs, defaultdict(TextReader, FROM_FORMAT))


# noinspection PyTypeChecker,PyStringFormat
def main():
    args = Config().args
    assert args.passages or args.train, "Either passages or --train is required (use -h for help)"
    assert args.model or args.train or args.folds, "Either --model or --train or --folds is required"
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
            print("Average labeled test F1 score for each fold: " + ", ".join(
                "%.3f" % s.average_f1() for s in fold_scores))
            print("Aggregated scores across folds:\n")
            scores.print()
            yield scores
    else:  # Simple train/dev/test by given arguments
        train_passages, dev_passages, test_passages = [read_passages(args, arg) for arg in
                                                       (args.train, args.dev, args.passages)]
        yield from train_test(train_passages, dev_passages, test_passages, args)


# def tracefunc(frame, event, arg):
#     if event.endswith("call") and arg:
#         if (getattr(arg, "__module__", None) or getattr(arg.__self__.__class__, "__module__")) == "_dynet":
#             print(">", os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, arg.__qualname__, "(",
#                   ", ".join("%s=%r" % (v, frame.f_locals[v])
#                             for v in frame.f_code.co_varnames[:frame.f_code.co_argcount] if v != "self"), ")")
#             # print(arg.__qualname__)
#     return tracefunc
# import sys
# sys.setprofile(tracefunc)

if __name__ == "__main__":
    list(main())
