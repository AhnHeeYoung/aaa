from xml.etree.ElementTree import parse
import numpy as np
import os


class Annotation():
    def __init__(self):
        self.Class = []
        self.x_s = []
        self.y_s = []
        self.points = []
        self.min_x = []
        self.max_x = []
        self.min_y = []
        self.max_y = []


def Parsing_Qupath_Annotation(annotation_fn, annot_c):
    def Points_Split(temp_points_str):
        temp_points_str = temp_points_str.replace('[', '')
        temp_points_str = temp_points_str.replace(']', '')
        temp_points_str = temp_points_str.split('Point:')[1:]
        return temp_points_str

    lines = [line.rstrip('\n') for line in open(annotation_fn)]
    img_path = lines[0]

    for i in range(1, len(lines), 2):
        annot_c.Class.append(lines[i])
        temp_points_str = Points_Split(lines[i + 1])

        temp_x_s = []
        temp_y_s = []
        temp_points = []
        for ii in range(0, len(temp_points_str)):
            temp_x = round(float(temp_points_str[ii].split(',')[0]))
            temp_y = round(float(temp_points_str[ii].split(',')[1]))
            temp_x_s.append(temp_x)
            temp_y_s.append(temp_y)
            temp_points.append([temp_x, temp_y])

        annot_c.x_s.append(temp_x_s)
        annot_c.y_s.append(temp_y_s)
        annot_c.min_x.append(min(temp_x_s))
        annot_c.max_x.append(max(temp_x_s))
        annot_c.min_y.append(min(temp_y_s))
        annot_c.max_y.append(max(temp_y_s))
        annot_c.points.append(temp_points)

    return annot_c, img_path


if __name__ == "__main__":
    # annot_path = 'E:\\Project\\Samsung_GC\\Data\\Samsung_Examples\\1_001.xml'
    # Annotation_X, Annotation_Y, class_id = xml_aperio_multi_class(annot_path)
    WSI_Annot = Annotation()
    annot_path = r'annotation/polygons.txt'
    annot_c, img_path = Parsing_Qupath_Annotation(annot_path, WSI_Annot)
    annot_c.Class[0]
    annot_c.points
    annot_c.min_x[0]
    annot_c.min_y[0]
    annot_c.max_x[0]
    annot_c.max_y[0]

    print(annot_c, img_path)