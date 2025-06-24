import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.k_nearest_neighbors import KNearestNeighbors

knn_model = KNearestNeighbors(file_path="object_classification_dataset\\pokemon_data.npz", X_label='images', y_label='labels')
# knn_model._model_machine_learning_pipeline(1)
print(knn_model.get_accuracy_score())
print(knn_model.predict("test2.jpg"))
knn_model.generate_confusion_matrix()
knn_model.save_model("trained_models\\pokemon_classifier_model.pkl")
KNearestNeighbors.load_model("trained_models\\pokemon_classifier_model.pkl")