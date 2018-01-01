from configargparse import ArgParser
from tqdm import tqdm
from ucca import layer1
from ucca.ioutil import read_files_and_dirs

argparser = ArgParser()
argparser.add_argument("dir")
argparser.add_argument("-v", "--verbose", action="store_true")
args = argparser.parse_args()

all_tags = {}
childful = {}
for p in tqdm(read_files_and_dirs(args.dir)):
    for n in p.layer(layer1.LAYER_ID).all:
        for e in n:
            all_tags[e.tag] = n
            if any(isinstance(x, layer1.FoundationalNode) for x in e.child.children):
                childful[e.tag] = n

print("All tags: %d" % len(all_tags))
childless = set(all_tags).difference(childful)
print("Childless: %d (%s)" % (len(childless), ", ".join(childless)))

if args.verbose:
    print("\n".join("%s: %s" % (tag, node) for tag, node in childful.items()))
