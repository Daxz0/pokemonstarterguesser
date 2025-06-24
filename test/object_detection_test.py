import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.draw_bounding_boxes_with_knn import draw_bounding_boxes_with_knn

draw_bounding_boxes_with_knn('test_files\\test4.webp')