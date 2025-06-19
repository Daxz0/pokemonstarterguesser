import cv2
import os

class DetectPokemonObjects:
    def __init__(self, image, model):
        self.image = image
        self.model = model

    def show_detected_image(self):
        image = cv2.imread("pokemon_scene.jpg")
        regions = self.define_pokemon_regions()
        classified_regions = self.classify_pokemon_regions(regions)
        output = self.draw_results(classified_regions)

        cv2.imshow("Detected Pokémon", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def define_pokemon_regions(self):
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        threshold_value, threshold_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > 30 and h > 30):
                regions.append(x, y, w, h)

        return regions

    def classify_pokemon_regions(self, regions):
        results = []

        counter = 0
        for (x, y, w, h) in regions:
            roi = self.image[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (64, 64)).flatten() / 255.0
            prediction = self.model.get_y_pred([roi_resized])[0]
            results.append((x, y, w, h, prediction[counter]))
            counter += 1

        return results

    def draw_results(self, results):
        for (x, y, w, h, label) in results:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(self.image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return self.image

file_path = r"C:\Data\Anish\Github\pokemonstarterguesser\data\bulbasaur\00000000.png"
print(os.path.exists(file_path))

# Load the image
image = cv2.imread(r"C:\Data\Anish\Github\pokemonstarterguesser\data\bulbasaur\00000000.png") # Replace with your image path

# Define bounding box coordinates
x1, y1 = 100, 200
x2, y2 = 200, 300

# Define color (BGR) and thickness
color = (0, 255, 0) # Green
thickness = 2

# Draw the rectangle
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Display the image
cv2.imshow('Image with Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image
cv2.imwrite('image_with_bbox.jpg', image)