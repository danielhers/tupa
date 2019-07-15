import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict, Counter
from enum import Enum
from itertools import chain

import score
from main import read_graphs
from tqdm import tqdm

from tupa.__version__ import GIT_VERSION
from tupa.config import Config, Iterations
from tupa.model import Model, NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY, SHARED_OUTPUT_KEYS
from tupa.oracle import Oracle
from tupa.states.state import State


class ParserException(Exception):
    pass


class ParseMode(Enum):
    train = 1
    dev = 2
    test = 3


class AbstractParser:
    def __init__(self, config, model, training=False, evaluation=False):
        self.config = config
        self.model = model
        self.training = training
        self.evaluation = evaluation
        self.count = Counter()
        self.correct_count = Counter()
        self.num_tokens = self.f1 = 0
        self.started = time.time()

    @property
    def duration(self):
        return (time.time() - self.started) or 1.0

    def tokens_per_second(self):
        return self.num_tokens / self.duration


class GraphParser(AbstractParser):
    """ Parser for a single graph, has a state and optionally an oracle """
    def __init__(self, graph, *args, conllu=None, alignment=None, target=None, **kwargs):
        """
        :param graph: gold Graph to get the correct nodes and edges from (in training), or just to get id from (in test)
        :param conllu: Graph with node per token predicted by a syntactic parser
        :param alignment: Graph with node.id corresponding to graph node id, node.label to conllu node.id (node align),
                          node properties to graph edge labs, and node property values to conllu node.id (edge align)
        :param target: framework to parse to
        """
        super().__init__(*args, **kwargs)
        self.graph, self.overlay = graph
        self.conllu = conllu
        self.alignment = alignment
        self.out = self.graph
        self.framework = target or self.graph.framework
        if self.config.args.bert_multilingual is not None:
            self.lang = self.graph.lang
            assert self.lang, "Attribute 'lang' is required per passage when using multilingual BERT"
        self.state_hash_history = set()
        self.state = self.oracle = None
        if self.framework == "amr":
            if self.alignment:  # Copy alignments to anchors, updating graph
                for alignment_node in self.alignment.nodes:
                    node = self.graph.find_node(alignment_node.id)
                    if node.anchors is None:
                        node.anchors = []
                    for conllu_node_id in (alignment_node.label or []) + list(chain(*alignment_node.values or [])):
                        conllu_node = conllu.find_node(conllu_node_id)
                        if conllu_node is None:
                            raise ValueError("Alignments incompatible with tokenization: token %s "
                                             "not found in graph %s" % (conllu_node_id, self.graph.id))
                        node.anchors += conllu_node.anchors
            else:
                raise ValueError("No alignment found for AMR graph " + self.graph.id)

    def init(self):
        self.config.set_framework(self.framework)
        self.state = State(self.graph, self.conllu, self.framework)
        # Graph is considered labeled if there are any nodes in it
        if self.training or ((self.config.args.verbose > 1 or self.config.args.action_stats) and self.state.labeled):
            self.oracle = Oracle(self.graph, self.conllu)
        else:
            self.oracle = None
        self.model.init_model(self.framework)
        self.model.init_features(self.framework, self.state, self.training)

    def parse(self, display=True, write=False, accuracies=None):
        self.init()
        graph_id = self.graph.id
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.parse_internal).result(self.config.args.timeout)
            status = "(%d tokens/s)" % self.tokens_per_second()
        except ParserException as e:
            if self.training:
                raise
            self.config.log("graph %s: %s" % (graph_id, e))
            status = "(failed)"
        except concurrent.futures.TimeoutError:
            self.config.log("graph %s: timeout (%fs)" % (graph_id, self.config.args.timeout))
            status = "(timeout)"
        return self.finish(status, display=display, write=write, accuracies=accuracies)

    def parse_internal(self):
        """
        Internal method to parse a single graph.
        If training, use oracle to train on given graphs. Otherwise just parse with classifier.
        """
        self.config.print("  initial state: %s" % self.state)
        while True:
            if self.config.args.check_loops:
                self.check_loop()
            true_actions = self.get_true_actions()
            action, predicted_action = self.choose(true_actions)
            self.state.transition(action)
            need_label, label, predicted_label, true_label = self.select_node_label(action)
            need_property, property_value, predicted_property_value, true_property_value = \
                self.select_node_property_value(action)
            need_attribute, attribute_value, predicted_attribute_value, true_attribute_value = \
                self.select_edge_attribute_value(action)
            if self.config.args.action_stats:
                try:
                    with open(self.config.args.action_stats, "a") as f:
                        print(",".join(map(str, [predicted_action, action] + list(true_actions.values()))), file=f)
                except OSError:
                    pass
            self.config.print(lambda: "\n".join(["  predicted: %-15s true: %-15s taken: %-15s %s" % (
                predicted_action, "|".join(map(str, true_actions.values())), action, self.state) if self.oracle else
                                          "  action: %-15s %s" % (action, self.state)] + (
                ["  predicted label: %-9s true label: %s" % (predicted_label, true_label)
                 if self.oracle else "  label: %s" % label] if need_label else []
                ["  predicted property: %-9s true property: %s" % (predicted_property_value, true_property_value)
                    if self.oracle else "  property: %s" % property_value] if need_property else []
                ["  predicted attribute: %-9s true attribute: %s" % (predicted_attribute_value, true_attribute_value)
                    if self.oracle else "  attribute: %s" % attribute_value] if need_attribute else []
            ) + ["    " + l for l in self.state.log]))
            if self.state.finished:
                return  # action is Finish

    def get_true_actions(self):
        true_actions = {}
        if self.oracle:
            try:
                true_actions = self.oracle.get_actions(self.state, self.model.output_values(self.framework),
                                                       create=self.training)
            except (AttributeError, AssertionError) as e:
                if self.training:
                    raise ParserException("Error in getting action from oracle during training on "
                                          + self.graph.id) from e
        return true_actions

    def get_true_node_label(self, node):
        try:
            return self.oracle.get_node_label(self.state, node) if self.oracle else (None, None)
        except AssertionError as e:
            if self.training:
                raise ParserException("Error in getting node label from oracle during training on "
                                      + self.graph.id + ", node " + str(node)) from e
            return None, None

    def get_true_node_property_value(self, node):
        try:
            return self.oracle.get_node_property_value(self.state, node) if self.oracle else (None, None)
        except AssertionError as e:
            if self.training:
                raise ParserException("Error in getting node property from oracle during training on "
                                      + self.graph.id + ", node " + str(node)) from e
            return None, None

    def get_true_edge_attribute_value(self, edge):
        try:
            return self.oracle.get_edge_attribute_value(self.state, edge) if self.oracle else (None, None)
        except AssertionError as e:
            if self.training:
                raise ParserException("Error in getting edge attribute from oracle during training on "
                                      + self.graph.id + ", edge " + str(edge)) from e
            return None, None

    def select_node_label(self, action=None):
        true_label = label = predicted_label = None
        need_label = self.state.need_label  # Label action, requires a choice of label
        if need_label:
            true_label, raw_true_label = self.get_true_node_label(action or need_label)
            label, predicted_label = self.choose(true_label, NODE_LABEL_KEY, "node label")
            self.state.assign_node_label(raw_true_label if label == true_label else label)
        return need_label, label, predicted_label, true_label

    def select_node_property_value(self, action=None):
        true_property_value = property_value = predicted_property_value = None
        need_property = self.state.need_property  # Property action, requires a choice of property + value
        if need_property:
            true_property_value = self.get_true_node_property_value(action or need_property)
            property_value, predicted_property_value = self.choose(true_property_value, NODE_PROPERTY_KEY,
                                                                   "node property")
            self.state.assign_node_property_value(property_value)
        return need_property, property_value, predicted_property_value, true_property_value

    def select_edge_attribute_value(self, action=None):
        true_attribute_value = attribute_value = predicted_attribute_value = None
        need_attribute = self.state.need_attribute  # Attribute action, requires a choice of attribute + value
        if need_attribute:
            true_attribute_value = self.get_true_edge_attribute_value(action or need_attribute)
            attribute_value, predicted_attribute_value = self.choose(true_attribute_value, EDGE_ATTRIBUTE_KEY,
                                                                     "edge attribute")
            self.state.assign_edge_attribute_value(attribute_value)
        return need_attribute, attribute_value, predicted_attribute_value, true_attribute_value

    def choose(self, true, key=None, name="action"):
        outputs = self.model.output_values(self.framework, key)
        true_keys = None if key is None else ((outputs[true],) if self.oracle else ())  # Must be before score()
        scores, features = self.model.score(self.state, self.framework, key)
        self.config.print(lambda: "  %s scores: %s" % (name, tuple(zip(outputs.all, scores))), level=4)
        try:
            output = pred = self.predict(scores, outputs.all, self.state.is_valid_annotation(key))
        except StopIteration as e:
            raise ParserException("No valid %s available\n%s" % (name, self.oracle.log if self.oracle else "")) from e
        output, is_correct, true_keys, true_values = self.correct(key, output, pred, scores, true, true_keys)
        if self.training:
            assert not self.model.is_finalized, "Updating finalized model"
            self.model.classifier.update(
                features, axis=self.model.key(self.framework, key), true=true_keys,
                pred=outputs[pred] if key is not None else pred.id,
                importance=[self.config.args.swap_importance if a.is_swap else 1 for a in true_values] or None)
        self.model.classifier.finished_step(self.training)
        return output, pred

    def correct(self, key, output, pred, scores, true, true_keys):
        true_values = ()
        if key is None:  # action
            true_keys, true_values = map(list, zip(*true.items())) if true else (None, None)
            output = true.get(pred.id)
            is_correct = (output is not None)
            if not is_correct:
                output = true_values[scores[true_keys].argmax()] if self.training else pred
        else:
            is_correct = (output == true)
        if is_correct:
            self.correct_count[key] += 1
        elif key is not None and self.oracle and self.training:
            output = true
        self.count[key] += 1
        return output, is_correct, true_keys, true_values

    @staticmethod
    def predict(scores, values, is_valid=None):
        """
        Choose action/label based on classifier
        Usually the best action/label is valid, so max is enough to choose it in O(n) time
        Otherwise, sorts all the other scores to choose the best valid one in O(n lg n)
        :return: valid action/label with maximum probability according to classifier
        """
        return next(filter(is_valid, (values[i] for i in GraphParser.generate_descending(scores))))

    @staticmethod
    def generate_descending(scores):
        yield scores.argmax()
        yield from scores.argsort()[::-1]  # Contains the max, but otherwise items might be missed (different order)

    def finish(self, status, display=True, write=False, accuracies=None):
        self.model.classifier.finished_item(self.training)
        if not self.training:
            self.out = self.state.create_graph()
        if write:
            json.dump(self.out.encode(), self.config.args.output, indent=None, ensure_ascii=False)
        if display:
            self.config.print("%s%.3fs %s" % (self.accuracy_str, self.duration, status), level=1)
        if accuracies is not None:
            accuracies[self.graph.id] = self.correct_count[None] / self.count[None] if self.count[None] else 0
        return self.out

    @property
    def accuracy_str(self):
        if self.oracle and self.count[None]:
            accuracy_str = "a=%-14s" % percents_str(self.correct_count[None], self.count[None])

            if self.count[NODE_LABEL_KEY]:
                accuracy_str += " l=%-14s" % percents_str(self.correct_count[NODE_LABEL_KEY],
                                                          self.count[NODE_LABEL_KEY])
            if self.count[NODE_PROPERTY_KEY]:
                accuracy_str += " l=%-14s" % percents_str(self.correct_count[NODE_LABEL_KEY],
                                                          self.count[NODE_LABEL_KEY])
            return "%-33s" % accuracy_str
        return ""

    def check_loop(self):
        """
        Check if the current state has already occurred, indicating a loop
        """
        h = hash(self.state)
        assert h not in self.state_hash_history, \
            "\n".join(["Transition loop", self.state.str("\n")] + [self.oracle.str("\n")] if self.oracle else ())
        self.state_hash_history.add(h)

    @property
    def num_tokens(self):
        return len(set(self.state.terminals).difference(self.state.buffer))  # To count even incomplete parses

    @num_tokens.setter
    def num_tokens(self, _):
        pass


class BatchParser(AbstractParser):
    """ Parser for a single training iteration or single pass over dev/test graphs """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_per_framework = defaultdict(int)
        self.num_graphs = 0

    def parse(self, graphs, display=True, write=False, accuracies=None, conllu=None, alignment=None):
        graphs, total = generate_and_len(single_to_iter(graphs))
        pr_width = len(str(total))
        id_width = 1
        graphs = self.add_progress_bar(graphs, display=display)
        for i, (graph, overlay) in enumerate(graphs, start=1):
            for target in graph.targets() or [graph.framework]:
                parser = GraphParser((graph, overlay), self.config, self.model, self.training, self.evaluation,
                                     conllu=conllu[graph.id], alignment=alignment.get(graph.id), target=target)
                if self.config.args.verbose and display:
                    progress = "%3d%% %*d/%d" % (i / total * 100, pr_width, i, total) \
                        if total and i <= total else "%d" % i
                    id_width = max(id_width, len(str(graph.id)))
                    print("%s %-6s %-*s" % (progress, target, id_width, graph.id),
                          end=self.config.line_end)
                else:
                    graphs.set_description()
                    postfix = {target: graph.id}
                    if display:
                        postfix["|t/s|"] = self.tokens_per_second()
                        if self.correct_count[None]:
                            postfix["|a|"] = percents_str(self.correct_count[None], self.count[None], fraction=False)
                        for key in SHARED_OUTPUT_KEYS:
                            if self.correct_count[key]:
                                postfix["|" + key + "|"] = percents_str(self.correct_count[key], self.count[key],
                                                                        fraction=False)
                    graphs.set_postfix(**postfix)
                self.seen_per_framework[target] += 1
                if self.training and self.config.args.max_training_per_framework and \
                        self.seen_per_framework[target] > self.config.args.max_training_per_framework:
                    self.config.print("skipped", level=1)
                    continue
                yield parser.parse(display=display, write=write, accuracies=accuracies)
                self.update_counts(parser)
        if self.num_graphs and display:
            self.summary()

    def add_progress_bar(self, it, total=None, display=True):
        return it if self.config.args.verbose and display else tqdm(
            it, unit="graph", total=total, file=sys.stderr, desc="Initializing")

    def update_counts(self, parser):
        self.count.update(parser.count)
        self.correct_count.update(parser.correct_count)
        self.num_tokens += parser.num_tokens
        self.num_graphs += 1

    def summary(self):
        print("Parsed %d graphs" % self.num_graphs)
        if self.correct_count[None]:
            accuracy_str = percents_str(self.correct_count[None], self.count[None], "correct actions ")
            for key, title in (NODE_LABEL_KEY, "node labels"), (NODE_PROPERTY_KEY, "node properties"), \
                              (EDGE_ATTRIBUTE_KEY, "edge attributes"):
                if self.count[key]:
                    accuracy_str += ", " + percents_str(self.correct_count[key], self.count[key],
                                                        "correct " + title + " ")
            print("Overall %s" % accuracy_str)
        print("Total time: %.3fs (average time/graph: %.3fs, average tokens/s: %d)" % (
            self.duration, self.time_per_graph(),
            self.tokens_per_second()), flush=True)

    def time_per_graph(self):
        return self.duration / self.num_graphs


class Parser(AbstractParser):
    """ Main class to implement transition-based meaning representation parser """
    def __init__(self, model_file=None, config=None):
        super().__init__(config=config or Config(), model=Model(model_file or config.args.classifier))
        self.best_score = self.dev = self.test = self.iteration = self.epoch = self.batch = None
        self.trained = self.save_init = False
        self.accuracies = {}

    def train(self, graphs=None, dev=None, test=None, iterations=1, conllu=None, alignment=None):
        """
        Train parser on given graphs
        :param graphs: iterable of graphs to train on
        :param dev: iterable of graphs to tune on
        :param test: iterable of graphs that would be tested on after train finished
        :param iterations: iterable of Iterations objects whose i attributes are the number of iterations to perform
        :param conllu: dict of graph id to graph specifying conllu preprocessing
        :param alignment: dict of graph id to graph specifying alignment preprocessing
        """
        self.trained = True
        self.dev = dev
        self.test = test
        if graphs:
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
                    if self.config.args.curriculum and self.accuracies:
                        print("Sorting graphs by previous epoch accuracy...")
                        graphs = sorted(graphs, key=lambda p: self.accuracies.get(p.id, 0))
                    else:
                        self.config.random.shuffle(graphs)
                    if not sum(1 for _ in self.parse(graphs, mode=ParseMode.train, conllu=conllu, alignment=alignment)):
                        raise ParserException("Could not train on any graph")
                    yield self.eval_and_save(self.iteration == len(iterations) and self.epoch == end - 1,
                                             finished_epoch=True, conllu=conllu, alignment=alignment)
                print("Trained %d epochs" % (end - 1))
                if dev:
                    if self.iteration < len(iterations):
                        self.model.load(is_finalized=False)  # Load best model to prepare for next iteration
                    elif test:
                        self.model.load()  # Load best model to prepare for test
        else:  # No graphs to train on, just load model
            self.model.load()
            self.print_config()

    def init_train(self):
        try:
            self.model.load(is_finalized=False)
        except FileNotFoundError:
            print("not found, starting from untrained model.")
        self.print_config()
        self.best_score = self.model.classifier.best_score if self.model.classifier else 0

    def eval_and_save(self, last=False, finished_epoch=False, conllu=None, alignment=None):
        scores = None
        model = self.model
        # noinspection PyAttributeOutsideInit
        self.model = finalized = model.finalize(finished_epoch=finished_epoch)
        if self.dev:
            if not self.best_score:
                self.save(finalized)
            average_score, scores = self.eval(self.dev, ParseMode.dev, self.config.args.devscores,
                                              conllu=conllu, alignment=alignment)
            if average_score >= self.best_score:
                print("Better than previous best score (%.3f)" % self.best_score)
                finalized.classifier.best_score = average_score
                if self.best_score:
                    self.save(finalized)
                self.best_score = average_score
                if self.config.args.eval_test and self.test and self.test is not True:  # There are graphs to parse
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

    def eval(self, graphs, mode, scores_filename, display=True, conllu=None, alignment=None):
        print("Evaluating on %s graphs" % mode.name)
        out = self.parse(graphs, mode=mode, evaluate=True, display=display, conllu=conllu, alignment=alignment)
        try:
            results = score.mces.evaluate([g for g, _ in graphs], out)
        except (KeyError, ValueError) as e:
            raise ValueError("Failed evaluating graphs: \n" + "\n".join(json.dumps(
                g.encode(), indent=None, ensure_ascii=False) for g in out)) from e
        prefix = ".".join(map(str, [self.iteration, self.epoch] + (
            [self.batch] if self.config.args.save_every else [])))
        if display:
            print("Evaluation %s, average F1 score on %s: %.3f" % (prefix, mode.name, results["all"]["f"]))
        print_scores(results, scores_filename, prefix=prefix, prefix_title="iteration")
        return results["all"]["f"], out

    def parse(self, graphs, mode=ParseMode.test, evaluate=False, display=True, write=False,
              conllu=None, alignment=None):
        """
        Parse given graphs
        :param graphs: iterable of graphs to parse
        :param mode: ParseMode value.
                     If train, use oracle to train on given graphs.
                     Otherwise, just parse with classifier.
        :param evaluate: whether to evaluate parsed graphs with respect to given ones.
                           Only possible when given graphs are annotated.
        :param display: whether to display information on each parsed graph
        :param write: whether to write output graphs to file
        :param conllu: dict of graph id to graph specifying conllu preprocessing
        :param alignment: dict of graph id to graph specifying alignment preprocessing
        :return: generator of parsed graphs (or in train mode, the original ones),
                 or, if evaluation=True, of pairs of (Graph, Scores).
        """
        self.batch = 0
        assert mode in ParseMode, "Invalid parse mode: %s" % mode
        training = (mode is ParseMode.train)
        if not training and not self.trained:
            yield from self.train()  # Try to load model from file
        parser = BatchParser(self.config, self.model, training, mode if mode is ParseMode.dev else evaluate)
        for i, graph in enumerate(parser.parse(graphs, display=display, write=write, accuracies=self.accuracies,
                                               conllu=conllu, alignment=alignment),
                                  start=1):
            if training and self.config.args.save_every and i % self.config.args.save_every == 0:
                self.eval_and_save(conllu=conllu, alignment=alignment)
                self.batch += 1
            yield graph

    def print_config(self):
        self.config.print("tupa %s" % (self.model.config if self.model else self.config), level=0)


def train_test(train_graphs, dev_graphs, test_graphs, args, model_suffix=""):
    """
    Train and test parser on given graph
    :param train_graphs: graph to train on
    :param dev_graphs: graphs to evaluate on every iteration
    :param test_graphs: graphs to test on after training
    :param args: extra argument
    :param model_suffix: string to append to model filename before file extension
    :return: generator of Scores objects: dev scores for each training iteration (if given dev), and finally test scores
    """
    base, ext = os.path.splitext(args.model or args.classifier)
    model_file = base + model_suffix + ext
    conllu = {graph.id: graph for graph, _ in read_graphs_with_progress_bar(args.conllu)}
    alignment = {graph.id: graph for graph, _ in read_graphs_with_progress_bar(args.alignment)}
    p = Parser(model_file=model_file, config=Config())
    yield from filter(None, p.train(train_graphs, dev=dev_graphs, test=test_graphs, iterations=args.iterations,
                                    conllu=conllu, alignment=alignment))
    if test_graphs:
        if args.train or args.folds:
            print("Evaluating on test graphs")
        evaluate = args.evaluate or train_graphs
        out = p.parse(test_graphs, evaluate=evaluate, write=args.write, conllu=conllu, alignment=alignment)
        results = score.mces.evaluate([g for g, _ in test_graphs], out)
        print_scores(results, args.testscores)
        yield results


def percents_str(part, total, infix="", fraction=True):
    ret = "%d%%" % (100 * part / total)
    if fraction:
        ret += " %s(%d/%d)" % (infix, part, total)
    return ret


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, prefix=prefix + "." + str(k))
        else:
            yield k, v


def print_scores(results, filename, prefix=None, prefix_title=None):
    if filename:
        print_title = not os.path.exists(filename)
        if prefix_title is not None:
            results = dict(results)
            results[prefix_title] = prefix
        keys, values = list(zip(*sorted(flatten(results))))
        try:
            with open(filename, "a") as f:
                if print_title:
                    print(*keys, file=f, sep=",")
                print(*values, file=f, sep=",")
        except OSError:
            pass


def single_to_iter(it):
    return it if hasattr(it, "__iter__") else (it,)  # Single graph given


def generate_and_len(it):
    return it, (len(it) if hasattr(it, "__len__") else None)


# noinspection PyTypeChecker,PyStringFormat
def main_generator():
    args = Config().args
    assert args.model or args.train or args.folds, "Either --model or --train or --folds is required"
    assert not (args.train or args.dev) or not args.folds, "--train and --dev are incompatible with --folds"
    assert args.train or not args.dev, "--dev is only possible together with --train"
    if args.folds:
        fold_scores = []
        all_graphs = read_graphs_with_progress_bar(args.input)
        assert len(all_graphs) >= args.folds, \
            "%d folds are not possible with only %d graphs" % (args.folds, len(all_graphs))
        Config().random.shuffle(all_graphs)
        folds = [all_graphs[i::args.folds] for i in range(args.folds)]
        for i in range(args.folds):
            print("Fold %d of %d:" % (i + 1, args.folds))
            dev_graphs = folds[i]
            test_graphs = folds[(i + 1) % args.folds]
            train_graphs = [graph for fold in folds if fold is not dev_graphs and fold is not test_graphs
                            for graph in fold]
            s = list(train_test(train_graphs, dev_graphs, test_graphs, args, "_%d" % i))
            if s and s[-1] is not None:
                fold_scores.append(s[-1])
        if fold_scores:
            print("Average test F1 score for each fold: " + ", ".join("%.3f" % s["all"]["f"] for s in fold_scores))
            print("Aggregated scores across folds:\n")
            yield fold_scores
    elif args.train:  # Simple train/dev/test by given arguments
        train_graphs, dev_graphs = [read_graphs_with_progress_bar(arg) if arg else []
                                    for arg in (args.input, args.dev)]
        yield from train_test(train_graphs, dev_graphs, test_graphs=None, args=args)
    else:
        yield from train_test(train_graphs=None, dev_graphs=None,
                              test_graphs=read_graphs_with_progress_bar(args.input), args=args)


def read_graphs_with_progress_bar(fh, **kwargs):
    if fh is None:
        return ()
    return list(zip(*read_graphs(tqdm(fh, desc="Reading " + fh.name, unit=" graphs"), format="mrp", **kwargs)))


def main():
    print("TUPA version " + GIT_VERSION + " (MRP)")
    list(main_generator())


if __name__ == "__main__":
    main()
