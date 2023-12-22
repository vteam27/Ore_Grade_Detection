import joblib
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

image_url = "https://upload.wikimedia.org/wikipedia/commons/5/5d/Mother_Lode_Gold_OreHarvard_mine_quartz-gold_vein.jpg"


loaded_model = joblib.load('LR_final.joblib')

FeedData=[53.17, 10.35, 1108.170000, 441.197000, 396.862000, 9.286860, 1.5600]
feature_names = ["% Iron Feed", "% Silica Feed", "Starch Flow", "Amina Flow", "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density"]
data = pd.DataFrame([FeedData], columns=feature_names)

# input_data = [FeedData]
predictions = loaded_model.predict(data)[0]
print("Linear Regression Predictions (Silica Impurity):", predictions)


class  ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 64, 64)  
        self.fc2 = nn.Linear(64+1, 4)

    def forward(self, x,silica_impurity):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        combined_features = torch.cat((x, silica_impurity.unsqueeze(1)), dim=1)
        x = self.fc2(combined_features)
        return x
# Instantiate your model
image_cnn = ImageCNN()

# Load the pretrained weights from the .pth file
model_path = 'Final_model.pth'
image_cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
image_cnn.eval()  # Set the model to evaluation mode
img_path = "GRADE_A_5.jpg"
image = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])  

input_image = transform(image)
input_image = input_image.unsqueeze(0)

silica_imp=torch.Tensor([55])
with torch.no_grad():
    predicted_labels = image_cnn(input_image, silica_imp)

probabilities = F.softmax(predicted_labels, dim=1)

predicted_class_index = torch.argmax(probabilities, dim=1).item()

class_names = [ 'Grade A', 'Grade B', 'Grade C', 'Grade D']

print("Predicted Probabilities:")
probs=probabilities.squeeze().tolist()
for i, probability in enumerate(probs):
    print(f"{class_names[i]}: {probability:.2f}")

print(f"Predicted Grade: {class_names[predicted_class_index]} with Confidence : {100*probs[predicted_class_index]:.4f}%")