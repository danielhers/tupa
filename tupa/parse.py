import time
from enum import Enum

import os
from collections import defaultdict
from ucca import diffutil, ioutil, textutil, layer1, evaluation
from ucca.convert import from_text, to_text

from scheme.convert import FROM_FORMAT, TO_FORMAT, CONVERTERS
from scheme.evaluate import EVALUATORS, Scores
from scheme.util.amr import LABEL_ATTRIB, LABEL_SEPARATOR
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
        self.trained = False

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
                    self.model.load()
                except FileNotFoundError:
                    print("not found, starting from untrained model.")
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
            print("Average labeled F1 score on dev: %.3f" % average_score)
            print_scores(scores, self.args.devscores, prefix_title="iteration",
                         prefix=[self.iteration] + ([self.eval_index] if self.args.save_every else []))
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
        train = (mode is ParseMode.train)
        if not train and not self.trained:
            list(self.train())  # Try to load model from file
        passage_word = "sentence" if self.args.sentences else "paragraph" if self.args.paragraphs else "passage"
        self.total_actions = 0
        self.total_correct_actions = 0
        total_duration = 0
        total_tokens = 0
        passage_index = 0
        if not hasattr(passages, "__iter__"):  # Single passage given
            passages = (passages,)
        for passage_index, passage in enumerate(passages):
            edges, node_labels = map(any, zip(*[(n.outgoing, n.attrib.get(LABEL_ATTRIB))
                                                for n in passage.layer(layer1.LAYER_ID).all]))
            labeled = edges or node_labels  # Passage is considered labeled if there are any edges or node labels in it
            print("%s %-7s" % (passage_word, passage.ID), end=Config().line_end, flush=True)
            started = time.time()
            self.action_count = self.correct_action_count = self.label_count = self.correct_label_count = 0
            textutil.annotate(passage, verbose=self.args.verbose > 1)  # tag POS and parse dependencies
            Config().set_format(passage.extra.get("format") or "ucca", labeled)
            self.state = State(passage)
            self.state_hash_history = set()
            self.oracle = Oracle(passage) if train or (
                  self.args.verbose or Config().args.use_gold_node_labels) and labeled or self.args.verify else None
            self.model.init_model()
            if ClassifierProperty.require_init_features in self.model.get_classifier_properties():
                self.model.init_features(self.state, train)
            failed = False
            try:
                self.parse_passage(train)  # This is where the actual parsing takes place
            except ParserException as e:
                if train:
                    raise
                Config().log("%s %s: %s" % (passage_word, passage.ID, e))
                failed = True
            guessed = self.state.create_passage(verify=self.args.verify) if not train or self.args.verify else passage
            duration = time.time() - started
            total_duration += duration
            num_tokens = len(set(self.state.terminals).difference(self.state.buffer))
            total_tokens += num_tokens
            if self.oracle:  # We have an oracle to verify by
                if not failed and self.args.verify:
                    self.verify_passage(guessed, passage, train)
                if self.action_count:
                    accuracy_str = "%d%% (%d/%d)" % (100*self.correct_action_count/self.action_count,
                                                     self.correct_action_count, self.action_count)
                    if self.label_count:
                        accuracy_str += " %d%% (%d/%d)" % (100*self.correct_label_count/self.label_count,
                                                           self.correct_label_count, self.label_count)
                    print("%-30s" % accuracy_str, end=Config().line_end)
            print("%0.3fs" % duration, end="")
            print("%-15s" % (" (failed)" if failed else " (%d tokens/s)" % (num_tokens / duration)), end="")
            print(Config().line_end, end="")
            if self.oracle:
                print(Config().line_end, flush=True)
            self.model.classifier.finished_item(train)
            self.total_correct_actions += self.correct_action_count
            self.total_actions += self.action_count
            self.total_correct_labels += self.correct_label_count
            self.total_labels += self.label_count
            if train and self.args.save_every and (passage_index+1) % self.args.save_every == 0:
                self.eval_and_save()
                self.eval_index += 1
            yield (guessed, self.evaluate_passage(guessed, passage)) if evaluate else guessed

        if passages:
            print("Parsed %d %ss" % (passage_index+1, passage_word))
            if self.oracle and self.total_actions:
                accuracy_str = "%d%% correct actions (%d/%d)" % (100*self.total_correct_actions/self.total_actions,
                                                                 self.total_correct_actions, self.total_actions)
                if self.total_labels:
                    accuracy_str += ", %d%% correct labels (%d/%d)" % (100*self.total_correct_labels/self.total_labels,
                                                                       self.total_correct_labels, self.total_labels)
                print("Overall %s on %s" % (accuracy_str, mode.name))
            if total_duration:
                print("Total time: %.3fs (average time/%s: %.3fs, average tokens/s: %d)" % (
                    total_duration, passage_word, total_duration / (passage_index+1),
                    total_tokens / total_duration), flush=True)

    def parse_passage(self, train):
        """
        Internal method to parse a single passage
        :param train: use oracle to train on given passages, or just parse with classifier?
        """
        if self.args.verbose > 1:
            print("  initial state: %s" % self.state)
        while True:
            if self.args.check_loops:
                self.check_loop()
            features = self.model.feature_extractor.extract_features(self.state)
            true_actions = self.get_true_actions(train)
            action, predicted_action = self.choose_action(features, train, true_actions)
            try:
                self.state.transition(action)
            except AssertionError as e:
                raise ParserException("Invalid transition: %s %s" % (action, self.state)) from e
            if self.args.verbose > 1:
                if self.oracle:
                    print("  predicted: %-15s true: %-15s taken: %-15s %s" % (
                        predicted_action, "|".join(map(str, true_actions.values())), action, self.state))
                else:
                    print("  action: %-15s %s" % (action, self.state))
            if self.state.need_label:  # Label action that requires a choice of label
                true_label = self.get_true_label(action.orig_node)
                label, predicted_label = self.choose_label(features, train, true_label)
                self.state.label_node(label)
                if self.args.verbose > 1:
                    if self.oracle and not Config().args.use_gold_node_labels:
                        print("  predicted label: %-15s true label: %-15s" % (predicted_label, true_label))
                    else:
                        print("  label: %-15s" % label)
            self.model.classifier.finished_step(train)
            if self.args.verbose > 1:
                for line in self.state.log:
                    print("    " + line)
            if self.state.finished:
                return  # action is Finish (or early update is triggered)

    def get_true_actions(self, train):
        true_actions = {}
        if self.oracle:
            try:
                true_actions = self.oracle.get_actions(self.state, self.model.actions, create=train)
            except (AttributeError, AssertionError) as e:
                if train:
                    raise ParserException("Error in oracle during training") from e
        return true_actions

    def choose_action(self, features, train, true_actions):
        scores = self.model.classifier.score(features, axis=self.args.format)  # Returns NumPy array
        if self.args.verbose > 2:
            print("  action scores: " + ",".join(("%s: %g" % x for x in zip(self.model.actions.all, scores))))
        try:
            predicted_action = self.predict(scores, self.model.actions.all, self.state.is_valid_action)
        except StopIteration as e:
            raise ParserException("No valid action available\n%s" % (self.oracle.log if self.oracle else "")) from e
        action = true_actions.get(predicted_action.id)
        is_correct = (action is not None)
        if is_correct:
            self.correct_action_count += 1
        else:
            action = Config().random.choice(list(true_actions.values())) if train else predicted_action
        if train and not (is_correct and
                          ClassifierProperty.update_only_on_error in self.model.get_classifier_properties()):
            best_action = self.predict(scores[list(true_actions.keys())], list(true_actions.values()))
            self.model.classifier.update(features, axis=self.args.format, pred=predicted_action.id, true=best_action.id,
                                         importance=self.args.swap_importance if best_action.is_swap else 1)
        if train and not is_correct and self.args.early_update:
            self.state.finished = True
        self.action_count += 1
        return action, predicted_action

    def get_true_label(self, node):
        true_label = None
        if self.oracle:
            if node is not None:
                true_label = node.attrib.get(LABEL_ATTRIB)
            if true_label is not None:
                true_label, _, _ = true_label.partition(LABEL_SEPARATOR)
                if not self.state.is_valid_label(true_label):
                    raise ParserException("True label is invalid: %s %s" % (true_label, self.state))
        return true_label

    def choose_label(self, features, train, true_label):
        true_id = self.model.labels[true_label] if self.oracle else None  # Needs to happen before score()
        if Config().args.use_gold_node_labels:
            return true_label, true_label
        scores = self.model.classifier.score(features, axis=NODE_LABEL_KEY)
        if self.args.verbose > 2:
            print("  label scores: " + ",".join(("%s: %g" % x for x in zip(self.model.labels.all, scores))))
        label = predicted_label = self.predict(scores, self.model.labels.all, self.state.is_valid_label)
        if self.oracle:
            is_correct = (label == true_label)
            if is_correct:
                self.correct_label_count += 1
            if train and not (is_correct and
                              ClassifierProperty.update_only_on_error in self.model.get_classifier_properties()):
                self.model.classifier.update(features, axis=NODE_LABEL_KEY, pred=self.model.labels[label], true=true_id)
                label = true_label
        self.label_count += 1
        return label, predicted_label

    def predict(self, scores, values, is_valid=None):
        """
        Choose action/label based on classifier
        Usually the best action/label is valid, so max is enough to choose it in O(n) time
        Otherwise, sorts all the other scores to choose the best valid one in O(n lg n)
        :return: valid action/label with maximum probability according to classifier
        """
        return next(filter(is_valid, (values[i] for i in self.generate_descending(scores))))

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
        converters = CONVERTERS.get(ref.extra.get("format"))  # returns (input converter, output converter) tuple
        score = EVALUATORS.get(ref.extra.get("format"), evaluation).evaluate(
            guessed, ref, converter=converters and converters[1],  # converter output is list of lines
            verbose=guessed and self.args.verbose > 2, constructions=self.args.constructions)
        print("F1=%.3f" % score.average_f1(), flush=True)
        return score

    def verify_passage(self, guessed, ref, show_diff):
        """
        Compare predicted passage to true passage and raise an exception if they differ
        :param ref: true passage
        :param guessed: predicted passage to compare
        :param show_diff: if passages differ, show the difference between them?
                          Depends on guessed having the original node IDs annotated in the "remarks" field for each node
        """
        assert ref.equals(guessed, ignore_node=self.ignore_node), \
            "Failed to produce true passage" + (diffutil.diff_passages(ref, guessed) if show_diff else "")


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
    print("%s %s" % (os.path.basename(__file__), Config()))
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
                print()
            if guessed_passage is not None and args.write:
                out_format = guessed_passage.extra.get("format", args.output_format)
                ioutil.write_passage(guessed_passage, output_format=out_format, binary=args.output_format == "pickle",
                                     outdir=args.outdir, prefix=args.prefix,
                                     converter=TO_FORMAT.get(out_format, Config().output_converter or to_text))
        if passage_scores:
            scores = Scores(passage_scores)
            if args.verbose <= 1 or len(passage_scores) > 1:
                print("\nAverage labeled F1 score on test: %.3f" % scores.average_f1())
                print("Aggregated scores:")
                scores.print()
            print_scores(scores, args.testscores)
            yield scores


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
                fields = [".".join(map(str, prefix))] + fields
            print(",".join(fields), file=f)


def read_passages(args, files):
    return ioutil.read_files_and_dirs(files, args.sentences, args.paragraphs,
                                      defaultdict(lambda: Config().input_converter or from_text, FROM_FORMAT))


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


if __name__ == "__main__":
    list(main())
