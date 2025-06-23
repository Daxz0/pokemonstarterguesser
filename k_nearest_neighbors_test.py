from k_nearest_neighbors import KNearestNeighbors

knn_model = KNearestNeighbors(file_path="pokemon_data.npz", X_label='images', y_label='labels')
# knn_model._model_machine_learning_pipeline(1)
print(knn_model.get_accuracy_score())
print(knn_model.predict("test2.jpg"))
knn_model.generate_confusion_matrix()
knn_model.save_model("pokemon_classifier_model.pkl")
KNearestNeighbors.load_model("pokemon_classifier_model.pkl")