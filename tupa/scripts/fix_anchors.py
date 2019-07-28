import argparse
import json
import os

from tqdm import tqdm

from tupa.states.anchors import expand_anchors, compress_anchors


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.filenames:
        basename = os.path.basename(f.name)
        with open(os.path.join(args.out_dir, basename), "w", encoding="utf-8") as out_f:
            for line in tqdm(list(f), unit=" graphs", desc=basename):
                try:
                    graph = json.loads(line)
                except json.decoder.JSONDecodeError:
                    continue
                nodes = graph.get("nodes")
                if nodes:
                    removed_ids = []
                    for node in nodes:
                        anchors = node.get("anchors") or ()
                        expanded = expand_anchors(anchors)
                        compressed = compress_anchors(expanded)
                        if compressed:
                            node["anchors"] = compressed
                        elif graph["framework"] == "eds":
                            removed_ids.append(node["id"])  # No anchoring found
                    nodes = [node for node in nodes if node["id"] not in removed_ids]
                    if nodes:
                        graph["nodes"] = nodes
                    else:
                        del graph["nodes"]
                    edges = graph.get("edges")
                    if edges:
                        edges = [edge for edge in edges if edge["source"] not in removed_ids
                                 and edge["target"] not in removed_ids]
                        if edges:
                            graph["edges"] = edges
                        else:
                            del graph["edges"]
                    tops = graph.get("tops")
                    if tops:
                        tops = [top for top in tops if top not in removed_ids]
                        if tops:
                            graph["tops"] = tops
                        else:
                            del graph["tops"]
                json.dump(graph, out_f, indent=None, ensure_ascii=False)
                print(file=out_f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("filenames", type=argparse.FileType("r", encoding="utf-8"), nargs="+")
    argparser.add_argument("-o", "--out-dir", default=".")
    main(argparser.parse_args())
