
import os
import itertools
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

Configuration = ["Configuration", "Identity", "Species", "Gestalt", "Possessor", "Whole", "Characteristic", "Possession",
                 "PartPortion", "Stuff", "Accompanier", "InsteadOf", "ComparisonRef", "RateUnit", "Quantity",
                 "Approximator", "SocialRel", "OrgRole"]
Participant = ["Participant", "Causer", "Agent", "Co-Agent", "Theme", "Co-Theme", "Topic", "Stimulus", "Experiencer",
               "Originator", "Recipient", "Cost", "Beneficiary", "Instrument"]
Circumstance = ["Circumstance", "Temporal", "Time", "StartTime", "EndTime", "Frequency", "Duration", "Interval",
                "Locus", "Source", "Goal", "Path", "Direction", "Extent", "Means", "Manner", "Explanation",
                "Purpose"]
Other = ["NAP", "A"]
SNACS = list(itertools.chain(Configuration, Participant, Circumstance, Other))


def add_categories(edge_elem):
    # <category layer_name="tokenization" parent_name="" slot="1" tag="Terminal" />
    tag = edge_elem.attrib["type"]
    if tag in SNACS:
        parent_category_elem = ET.SubElement(edge_elem, 'category', layer_name="foundational layer",
                                             parent_name='', slot="3", tag="A")
        refinement_category_elem = ET.SubElement(edge_elem, 'category', layer_name="semantic role snacs",
                                                 parent_name='A', slot="1", tag=tag)
    else:
        category = ET.SubElement(edge_elem, 'category', layer_name="foundational layer",
                                 parent_name='', slot="3", tag=tag)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def convert_file(xml_file):
    tree = ET.parse(xml_file)
    for edge_element in tree.iter("edge"):
        add_categories(edge_element)
    with open(xml_file, "w") as f:
        f.write(prettify(tree.getroot()))


def remove_new_lines(file_name):
    with open(file_name, "r") as f:
        file_txt = f.read()
        file_txt = file_txt.replace("    ", "")
    with open(file_name, "w") as f:
        f.write(file_txt)


def remove_extra_new_lines(file_name):
    with open(file_name, "r") as f:
        file_txt = f.read()
        file_list = file_txt.split("\n")
        new_file_list = []
        for x in file_list:
            if x.replace("\t", "").replace(" ", "") == "":
                continue
            else:
                new_file_list.append(x)
        file_txt = "\n".join(new_file_list)
    with open(file_name, "w") as f:
        f.write(file_txt)


def edit_files(dir):
    for xml_file in os.listdir(dir):
        if not xml_file.endswith(".xml"):
            continue
        #remove_new_lines(os.path.join(dir, xml_file))
        #convert_file(os.path.join(dir, xml_file))
        remove_extra_new_lines(os.path.join(dir, xml_file))

if __name__ == "__main__":
    edit_files("./wiki-sentences-participants")
    edit_files("./wiki-sentences-participants/dev")
    edit_files("./wiki-sentences-participants/test")
    edit_files("./wiki-sentences-participants/train")