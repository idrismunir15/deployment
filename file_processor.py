from xml.etree.ElementTree import ElementTree
import pandas as pd
import glob
import os


def ecg_parse(xml_file):
    """
    This code is used to parse the xml file and return the data in a dictionary
    The process involves the following steps:
    
    1. Parse the xml file
    2. Extract the data from the xml file
    3. Return the data in a dictionary
    4. The dictionary is used to create a dataframe
    5. The dataframe is used to create a csv file

    Args:
        file (xml): The xml file to be parsed

    Returns:
        dictionary : The features extracted from the xml file
    """
    
    tree = ElementTree()
    tree.parse(xml_file)

    data={}
    for observation in tree.findall("VitalSigns/TwelveLeadReport/Observations/Observation")[:-2]:
        try:
            data.update({f"{observation.attrib['Type']}_{observation.attrib['Lead']}":observation.text})
        except:
            data.update({f"{observation.attrib['Type']}":observation.text})
    

    demograph=  tree.find("Patient/Demographics")

    demographics={}
    for ele in demograph:
        value=''
        if ele.tag in ['Age','Gender']:
            demographics[ele.tag]=ele.text
        else:
            pass
    
    return {**demographics,**data}


