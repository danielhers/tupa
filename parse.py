import os
import time
from enum import Enum

from classifiers.classifier import ClassifierProperty
from parsing.action import Actions
from parsing.config import Config
from parsing.model import Model
from parsing.oracle import Oracle
from states.state import State
from ucca import diffutil, ioutil, textutil, evaluation, layer0, layer1


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
        self.state = None  # State object created at each parse
        self.oracle = None  # Oracle object created at each parse
        self.scores = None  # NumPy array of action scores at each action
        self.action_count = 0
        self.correct_count = 0
        self.total_actions = 0
        self.total_correct = 0
        self.model = Model(model_type, model_file, Actions().all)
        self.beam = beam  # Currently unused
        self.state_hash_history = None  # For loop checking
        # Used in verify_passage to optionally ignore a mismatch in linkage nodes:
        self.ignore_node = None if Config().args.linkage else lambda n: n.tag == layer1.NodeTags.Linkage
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
            if Config().args.devscores:
                with open(Config().args.devscores, "w") as f:
                    print(",".join(["iteration"] + evaluation.Scores.field_titles(Config().args.constructions)), file=f)
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
            scores = evaluation.Scores.aggregate(scores)
            self.dev_scores.append(scores)
            score = scores.average_f1()
            print("Average labeled F1 score on dev: %.3f" % score)
            if Config().args.devscores:
                prefix = [self.iteration]
                if Config().args.save_every:
                    prefix.append(self.eval_index)
                with open(Config().args.devscores, "a") as f:
                    print(",".join([".".join(map(str, prefix))] + scores.fields()), file=f)
            if score >= self.best_score:
                print("Better than previous best score (%.3f)" % self.best_score)
                self.best_score = score
                self.model.save()
            else:
                print("Not better than previous best score (%.3f)" % self.best_score)
        elif last or Config().args.save_every is not None:
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
        passage_word = "sentence" if Config().args.sentences else \
                       "paragraph" if Config().args.paragraphs else \
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
            textutil.annotate(passage, verbose=Config().args.verbose)  # tag POS and parse dependencies
            self.state = State(passage)
            self.state_hash_history = set()
            self.oracle = Oracle(passage) if train else None
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
            predicted_passage = self.state.create_passage(assert_proper=Config().args.verify) \
                if not train or Config().args.verify else passage
            duration = time.time() - started
            total_duration += duration
            num_tokens -= len(self.state.buffer)
            total_tokens += num_tokens
            if train:  # We have an oracle to verify by
                if not failed and Config().args.verify:
                    self.verify_passage(passage, predicted_passage, train)
                if self.action_count:
                    print("%-16s" % ("%d%% (%d/%d)" %
                          (100 * self.correct_count / self.action_count,
                           self.correct_count, self.action_count)), end=Config().line_end)
            print("%0.3fs" % duration, end="")
            print("%-15s" % (" (failed)" if failed else " (%d tokens/s)" % (num_tokens / duration)), end="")
            print(Config().line_end, end="")
            if train:
                print(Config().line_end, flush=True)
            self.model.model.finished_item(train=train)
            self.total_correct += self.correct_count
            self.total_actions += self.action_count
            if train and Config().args.save_every and (passage_index+1) % Config().args.save_every == 0:
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
        if Config().args.verbose:
            print("  initial state: %s" % self.state)
        while True:
            if Config().args.check_loops:
                self.check_loop(print_oracle=train)

            true_actions = []
            if self.oracle is not None:
                try:
                    true_actions = self.oracle.get_actions(self.state)
                except (AttributeError, AssertionError) as e:
                    if train:
                        raise ParserException("Error in oracle during training") from e

            features = self.model.feature_extractor.extract_features(self.state)
            predicted_action = self.predict_action(features, true_actions)  # sets self.scores
            action = predicted_action
            correct_action = False
            if not true_actions:
                true_actions = "?"
            elif predicted_action in true_actions:
                self.correct_count += 1
                correct_action = True
            elif train:
                action = Config().random.choice(true_actions)
            if train and not (correct_action and
                              ClassifierProperty.update_only_on_error in self.model.model.get_classifier_properties()):
                best_true_action = true_actions[0] if len(true_actions) == 1 else \
                    true_actions[self.scores[[a.id for a in true_actions]].argmax()]
                self.model.model.update(features, predicted_action.id, best_true_action.id,
                                        Config().args.swap_importance if best_true_action.is_swap else 1)
            self.action_count += 1
            self.model.model.finished_step(train=train)
            try:
                self.state.transition(action)
            except AssertionError as e:
                raise ParserException("Invalid transition (%s): %s" % (action, e)) from e
            if Config().args.verbose:
                if self.oracle is None:
                    print("  action: %-15s %s" % (action, self.state))
                else:
                    print("  predicted: %-15s true: %-15s taken: %-15s %s" % (
                        predicted_action, "|".join(map(str, true_actions)), action, self.state))
                for line in self.state.log:
                    print("    " + line)
            if self.state.finished or train and not correct_action and Config().args.early_update:
                return  # action is FINISH

    def check_loop(self, print_oracle):
        """
        Check if the current state has already occurred, indicating a loop
        :param print_oracle: whether to print the oracle in case of an assertion error
        """
        h = hash(self.state)
        assert h not in self.state_hash_history,\
            "\n".join(["Transition loop", self.state.str("\n")] +
                      [self.oracle.str("\n")] if print_oracle else ())
        self.state_hash_history.add(h)

    def predict_action(self, features, true_actions):
        """
        Choose action based on classifier
        :param features: extracted feature values
        :param true_actions: from the oracle, to copy orig_node if the same action is selected
        :return: valid action with maximum probability according to classifier
        """
        self.scores = self.model.model.score(features)  # Returns a NumPy array
        if Config().args.verbose >= 2:
            print("  scores: " + " ".join(("%g" % s for s in self.scores)))
        best_action = self.select_action(self.scores.argmax(), true_actions)
        if self.state.is_valid(best_action):
            return best_action
        # Usually the best action is valid, so max is enough to choose it in O(n) time
        # Otherwise, sort all the other scores to choose the best valid one in O(n lg n)
        sorted_ids = self.scores.argsort()[::-1]
        actions = (self.select_action(i, true_actions) for i in sorted_ids)
        try:
            return next(a for a in actions if self.state.is_valid(a))
        except StopIteration as e:
            raise ParserException("No valid actions available\n" +
                                  ("True actions: %s" % true_actions if true_actions
                                   else self.oracle.log if self.oracle is not None
                                   else "") +
                                  "\nReturned actions: %s" %
                                  [self.select_action(i) for i in sorted_ids] +
                                  "\nScores: %s" % self.scores
                                  ) from e

    @staticmethod
    def select_action(i, true_actions=()):
        """
        Find action with the given ID in true actions (if exists) or in all actions
        :param i: ID to lookup
        :param true_actions: preferred set of actions to look in first
        :return: Action with id=i
        """
        try:
            return next(a for a in true_actions if a.id == i)
        except StopIteration:
            return Actions().all[i]

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
                ioutil.write_passage(guessed_passage, args)
        if passage_scores and (not args.verbose or len(passage_scores) > 1):
            test_scores = evaluation.Scores.aggregate(passage_scores)
            print("\nAverage labeled F1 score on test: %.3f" % test_scores.average_f1())
            print("Aggregated scores:")
            test_scores.print()
            if Config().args.testscores:
                with open(Config().args.testscores, "a") as f:
                    print(",".join(test_scores.fields()), file=f)
    return test_scores, p.dev_scores


def evaluate_passage(guessed_passage, ref_passage):
    score = evaluation.evaluate(guessed_passage, ref_passage, constructions=Config().args.constructions,
                                verbose=Config().args.verbose and guessed_passage is not None)
    print("F1=%.3f" % score.average_f1(), flush=True)
    return score


def main():
    args = Config().args
    assert args.passages or args.train, "Either passages or --train is required (use -h for help)"
    assert args.model or args.train or args.folds, "Either --model or --train or --folds is required"
    assert not (args.train or args.dev) or args.folds is None, "--train and --dev are incompatible with --folds"
    assert args.train or not args.dev, "--dev is only possible together with --train"
    print("Running parser with %s" % Config())
    test_scores = None
    dev_scores = None
    if Config().args.testscores:
        with open(Config().args.testscores, "w") as f:
            print(",".join(evaluation.Scores.field_titles(Config().args.constructions)), file=f)
    if args.folds is not None:
        k = args.folds
        fold_scores = []
        all_passages = list(ioutil.read_files_and_dirs(args.passages,
                                                       Config().args.sentences, Config().args.paragraphs))
        assert len(all_passages) >= k,\
            "%d folds are not possible with only %d passages" % (k, len(all_passages))
        Config().random.shuffle(all_passages)
        folds = [all_passages[i::k] for i in range(k)]
        for i in range(k):
            print("Fold %d of %d:" % (i + 1, k))
            dev_passages = folds[i]
            test_passages = folds[(i+1) % k]
            train_passages = [passage for fold in folds
                              if fold is not dev_passages and fold is not test_passages
                              for passage in fold]
            s, _ = train_test(train_passages, dev_passages, test_passages, args, "_%d" % i)
            if s is not None:
                fold_scores.append(s)
        if fold_scores:
            test_scores = evaluation.Scores.aggregate(fold_scores)
            print("Average labeled test F1 score for each fold: " + ", ".join(
                "%.3f" % s.average_f1() for s in fold_scores))
            print("Aggregated scores across folds:\n")
            test_scores.print()
    else:  # Simple train/dev/test by given arguments
        train_passages, dev_passages, test_passages = [ioutil.read_files_and_dirs(arg,
                                                       Config().args.sentences, Config().args.paragraphs) for arg in
                                                       (args.train, args.dev, args.passages)]
        test_scores, dev_scores = train_test(train_passages, dev_passages, test_passages, args)
    return test_scores, dev_scores


if __name__ == "__main__":
    main()
    Config().close()
