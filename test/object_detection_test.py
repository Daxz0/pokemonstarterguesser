import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.Constants as Constants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.draw_bounding_boxes_with_knn import draw_bounding_boxes_with_knn

draw_bounding_boxes_with_knn(Constants.TEST_FILES_PATH + '\\test4.webp')