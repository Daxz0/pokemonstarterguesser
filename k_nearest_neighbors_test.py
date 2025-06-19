from k_nearest_neighbors import KNearestNeighbors

knn_model = KNearestNeighbors(file_path="pokemon_data.npz", X_label='images', y_label='labels')
knn_model._model_machine_learning_pipeline(1)
print(knn_model.return_accuracy_score())
print(knn_model.predict("C:\\Data\\Anish\\Github\\pokemonstarterguesser\\converted_data\\bulbasaur\\00000001.png"))
knn_model.generate_confusion_matrix()