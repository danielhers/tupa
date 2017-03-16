import os
import time
from enum import Enum

from classifiers.classifier import ClassifierProperty
from states.state import State
from tupa.action import Actions
from tupa.config import Config
from tupa.model import Model, ACTION_AXIS, LABEL_AXIS
from tupa.oracle import Oracle
from ucca import diffutil, ioutil, textutil, layer0, layer1


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
        self.state = None  # State object created at each parse
        self.oracle = None  # Oracle object created at each parse
        self.action_count = 0
        self.correct_count = 0
        self.total_actions = 0
        self.total_correct = 0
        self.model = Model(model_type, model_file)
        self.update_only_on_error = \
            ClassifierProperty.update_only_on_error in self.model.model.get_classifier_properties()
        self.beam = beam  # Currently unused
        self.state_hash_history = None  # For loop checking
        # Used in verify_passage to optionally ignore a mismatch in linkage nodes:
        self.ignore_node = None if self.args.linkage else lambda n: n.tag == layer1.NodeTags.Linkage
        self.best_score = self.dev = self.iteration = self.eval_index = None
        self.dev_scores = []
        self.trained = False

    def train(self, passages=None, dev=None, iterations=1):
        """
        Train parser on given passages
        :param passages: iterable of passages to train on
        :param dev: iterable of passages to tune on
        :param iterations: number of iterations to perform
        :return: trained model
        """
        self.trained = True
        if passages:
            if ClassifierProperty.trainable_after_saving in self.model.model.get_classifier_properties():
                try:
                    self.model.load()
                except FileNotFoundError:
                    print("not found, starting from untrained model.")
            self.best_score = 0
            self.dev = dev
            if self.args.devscores:
                with open(self.args.devscores, "w") as f:
                    print(",".join(["iteration"] + Config().Scores.field_titles(self.args.constructions)), file=f)
            for self.iteration in range(1, iterations + 1):
                self.eval_index = 0
                print("Training iteration %d of %d: " % (self.iteration, iterations))
                list(self.parse(passages, mode=ParseMode.train))
                self.eval_and_save(self.iteration == iterations, finished_epoch=True)
                Config().random.shuffle(passages)
            print("Trained %d iterations" % iterations)
        if dev or not passages:
            self.model.load()

    def eval_and_save(self, last=False, finished_epoch=False):
        model = self.model
        self.model = self.model.finalize(finished_epoch=finished_epoch)
        if self.dev:
            print("Evaluating on dev passages")
            scores = [s for _, s in self.parse(self.dev, mode=ParseMode.dev, evaluate=True)]
            scores = Config().Scores.aggregate(scores)
            self.dev_scores.append(scores)
            score = scores.average_f1()
            print("Average labeled F1 score on dev: %.3f" % score)
            if self.args.devscores:
                prefix = [self.iteration]
                if self.args.save_every:
                    prefix.append(self.eval_index)
                with open(self.args.devscores, "a") as f:
                    print(",".join([".".join(map(str, prefix))] + scores.fields()), file=f)
            if score >= self.best_score:
                print("Better than previous best score (%.3f)" % self.best_score)
                self.best_score = score
                self.model.save()
            else:
                print("Not better than previous best score (%.3f)" % self.best_score)
        elif last or self.args.save_every is not None:
            self.model.save()
        if not last:
            self.model = model  # Restore non-finalized model

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
            self.train()
        passage_word = "sentence" if self.args.sentences else \
                       "paragraph" if self.args.paragraphs else \
                       "passage"
        self.total_actions = 0
        self.total_correct = 0
        total_duration = 0
        total_tokens = 0
        passage_index = 0
        if not hasattr(passages, "__iter__"):  # Single passage given
            passages = (passages,)
        for passage_index, passage in enumerate(passages):
            l0 = passage.layer(layer0.LAYER_ID)
            num_tokens = len(l0.all)
            l1 = passage.layer(layer1.LAYER_ID)
            labeled = len(l1.all) > 1
            assert not train or labeled, "Cannot train on unannotated passage: %s" % passage.ID
            assert not evaluate or labeled, "Cannot evaluate on unannotated passage: %s" % passage.ID
            print("%s %-7s" % (passage_word, passage.ID), end=Config().line_end, flush=True)
            started = time.time()
            self.action_count = 0
            self.correct_count = 0
            textutil.annotate(passage, verbose=self.args.verbose > 1)  # tag POS and parse dependencies
            self.state = State(passage)
            self.state_hash_history = set()
            self.oracle = Oracle(passage) if train or self.args.verbose and labeled else None
            failed = False
            if ClassifierProperty.require_init_features in self.model.model.get_classifier_properties():
                self.model.init_features(self.state, train)
            try:
                self.parse_passage(train)  # This is where the actual parsing takes place
            except ParserException as e:
                if train:
                    raise
                Config().log("%s %s: %s" % (passage_word, passage.ID, e))
                failed = True
            predicted_passage = self.state.create_passage(assert_proper=self.args.verify) \
                if not train or self.args.verify else passage
            duration = time.time() - started
            total_duration += duration
            num_tokens -= len(self.state.buffer)
            total_tokens += num_tokens
            if self.oracle:  # We have an oracle to verify by
                if not failed and self.args.verify:
                    self.verify_passage(passage, predicted_passage, train)
                if self.action_count:
                    print("%-16s" % ("%d%% (%d/%d)" %
                          (100 * self.correct_count / self.action_count,
                           self.correct_count, self.action_count)), end=Config().line_end)
            print("%0.3fs" % duration, end="")
            print("%-15s" % (" (failed)" if failed else " (%d tokens/s)" % (num_tokens / duration)), end="")
            print(Config().line_end, end="")
            if self.oracle:
                print(Config().line_end, flush=True)
            self.model.model.finished_item(train)
            self.total_correct += self.correct_count
            self.total_actions += self.action_count
            if train and self.args.save_every and (passage_index+1) % self.args.save_every == 0:
                self.eval_and_save()
                self.eval_index += 1
            yield (predicted_passage, evaluate_passage(predicted_passage, passage)) if evaluate else predicted_passage

        if passages:
            print("Parsed %d %ss" % (passage_index+1, passage_word))
            if self.oracle and self.total_actions:
                print("Overall %d%% correct transitions (%d/%d) on %s" %
                      (100 * self.total_correct / self.total_actions,
                       self.total_correct, self.total_actions,
                       mode.name))
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
            true_actions = self.get_oracle_actions(train)
            features = self.model.feature_extractor.extract_features(self.state)
            scores = self.model.model.score(features, axis=ACTION_AXIS)  # Returns NumPy array
            if self.args.verbose > 2:
                print("  action scores: " + ",".join(("%s: %g" % x for x in zip(Actions().all, scores))))
            try:
                predicted_action = self.predict(scores, Actions().all, self.state.is_valid_action)
            except StopIteration as e:
                raise ParserException("No valid action available\n%s" % (self.oracle.log if self.oracle else "")) from e
            action = true_actions.get(predicted_action.id)
            is_correct_action = (action is not None)
            if is_correct_action:
                self.correct_count += 1
            else:
                action = Config().random.choice(list(true_actions.values())) if train else predicted_action
            if train and not (is_correct_action and self.update_only_on_error):
                best_action = self.predict(scores[list(true_actions.keys())], list(true_actions.values()))
                self.model.model.update(features, axis=ACTION_AXIS, pred=predicted_action.id, true=best_action.id,
                                        importance=self.args.swap_importance if best_action.is_swap else 1)
            self.label_action(action, features, train)
            self.action_count += 1
            self.model.model.finished_step(train)
            try:
                self.state.transition(action)
            except AssertionError as e:
                raise ParserException("Invalid transition (%s): %s" % (action, e)) from e
            if self.args.verbose > 1:
                if self.oracle:
                    print("  predicted: %-15s true: %-15s taken: %-15s %s" % (
                        predicted_action, "|".join(map(str, true_actions.values())), action, self.state))
                else:
                    print("  action: %-15s %s" % (action, self.state))
                for line in self.state.log:
                    print("    " + line)
            if self.state.finished or train and not is_correct_action and self.args.early_update:
                return  # action is Finish

    def label_action(self, action, features, train):
        labels = self.model.labels
        if labels and action.has_label:  # Node-creating action that requires a label
            scores = self.model.model.score(features, axis=LABEL_AXIS)
            if self.args.verbose > 2:
                print("  label scores: " + ",".join(("%s: %g" % x for x in zip(labels.all, scores))))
            try:
                label = self.predict(scores, labels.all, self.state.is_valid_label)
            except StopIteration:
                label = None  # Empty label
            if train:
                if not (label == action.label and self.update_only_on_error):
                    self.model.model.update(features, axis=LABEL_AXIS, pred=labels[label], true=labels[action.label])
            else:
                action.label = label

    def get_oracle_actions(self, train):
        true_actions = {}
        if self.oracle:
            try:
                true_actions = self.oracle.get_actions(self.state)
            except (AttributeError, AssertionError) as e:
                if train:
                    raise ParserException("Error in oracle during training") from e
        return true_actions

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
        yield from scores.argsort()[::-1]

    def check_loop(self):
        """
        Check if the current state has already occurred, indicating a loop
        """
        h = hash(self.state)
        assert h not in self.state_hash_history,\
            "\n".join(["Transition loop", self.state.str("\n")] + [self.oracle.str("\n")] if self.oracle else ())
        self.state_hash_history.add(h)

    def verify_passage(self, passage, predicted_passage, show_diff):
        """
        Compare predicted passage to true passage and die if they differ
        :param passage: true passage
        :param predicted_passage: predicted passage to compare
        :param show_diff: if passages differ, show the difference between them?
                          Depends on predicted_passage having the original node IDs annotated
                          in the "remarks" field for each node.
        """
        assert passage.equals(predicted_passage, ignore_node=self.ignore_node),\
            "Failed to produce true passage" + \
            (diffutil.diff_passages(
                    passage, predicted_passage) if show_diff else "")


def train_test(train_passages, dev_passages, test_passages, args, model_suffix=""):
    """
    Train and test parser on given passage
    :param train_passages: passage to train on
    :param dev_passages: passages to evaluate on every iteration
    :param test_passages: passages to test on after training
    :param args: extra argument
    :param model_suffix: string to append to model filename before file extension
    :return: pair of (test scores, list of dev scores per iteration) where each one is a Scores object
    """
    test_scores = None
    model_base, model_ext = os.path.splitext(args.model or "ucca_" + args.classifier)
    p = Parser(model_file=model_base + model_suffix + model_ext, model_type=args.classifier, beam=args.beam)
    p.train(train_passages, dev=dev_passages, iterations=args.iterations)
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
            if guessed_passage is not None and not args.no_write:
                ioutil.write_passage(guessed_passage, output_format=args.output_format, binary=args.binary,
                                     outdir=args.outdir, prefix=args.prefix,
                                     default_converter=Config().output_converter)
        if passage_scores and (args.verbose <= 1 or len(passage_scores) > 1):
            test_scores = Config().Scores.aggregate(passage_scores)
            print("\nAverage labeled F1 score on test: %.3f" % test_scores.average_f1())
            print("Aggregated scores:")
            test_scores.print()
            if args.testscores:
                with open(args.testscores, "a") as f:
                    print(",".join(test_scores.fields()), file=f)
    return test_scores, p.dev_scores


def evaluate_passage(guessed, ref):
    score = Config().evaluate(
        guessed, ref,
        converter=None if Config().output_converter is None else lambda p: Config().output_converter(p)[0],
        verbose=Config().args.verbose > 2 and guessed is not None,
        constructions=Config().args.constructions)
    print("F1=%.3f" % score.average_f1(), flush=True)
    return score


# noinspection PyTypeChecker,PyStringFormat
def main():
    args = Config().args
    assert args.passages or args.train, "Either passages or --train is required (use -h for help)"
    assert args.model or args.train or args.folds, "Either --model or --train or --folds is required"
    assert not (args.train or args.dev) or args.folds is None, "--train and --dev are incompatible with --folds"
    assert args.train or not args.dev, "--dev is only possible together with --train"
    print("Running parser with %s" % Config())
    test_scores = None
    dev_scores = None
    if args.testscores:
        with open(args.testscores, "w") as f:
            print(",".join(Config().Scores.field_titles(args.constructions)), file=f)
    if args.folds is not None:
        fold_scores = []
        all_passages = list(ioutil.read_files_and_dirs(
            args.passages, args.sentences, args.paragraphs, Config().input_converter))
        assert len(all_passages) >= args.folds, \
            "%d folds are not possible with only %d passages" % (args.folds, len(all_passages))
        Config().random.shuffle(all_passages)
        folds = [all_passages[i::args.folds] for i in range(args.folds)]
        for i in range(args.folds):
            print("Fold %d of %d:" % (i + 1, args.folds))
            dev_passages = folds[i]
            test_passages = folds[(i + 1) % args.folds]
            train_passages = [passage for fold in folds
                              if fold is not dev_passages and fold is not test_passages
                              for passage in fold]
            s, _ = train_test(train_passages, dev_passages, test_passages, args, "_%d" % i)
            if s is not None:
                fold_scores.append(s)
        if fold_scores:
            test_scores = Config().Scores.aggregate(fold_scores)
            print("Average labeled test F1 score for each fold: " + ", ".join(
                "%.3f" % s.average_f1() for s in fold_scores))
            print("Aggregated scores across folds:\n")
            test_scores.print()
    else:  # Simple train/dev/test by given arguments
        train_passages, dev_passages, test_passages = [ioutil.read_files_and_dirs(
            arg, args.sentences, args.paragraphs, Config().input_converter) for arg in
                                                       (args.train, args.dev, args.passages)]
        test_scores, dev_scores = train_test(train_passages, dev_passages, test_passages, args)
    return test_scores, dev_scores


if __name__ == "__main__":
    main()
    Config().close()
