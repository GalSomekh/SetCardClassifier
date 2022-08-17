# SetCardClassifier
Python library for recognizing and labeling Set game cards in an image, video or camera stream.
## Setup
Install the required libraries:
```
pip install -r requirements.txt
```
## Example
Run the example 
```
python example.py 
```
## About the Project
This computer vision project was developed for an HCI (Human Computer Interaction) psychological experiment that was carried out by the [Milab](https://milab.runi.ac.il/) in the Reichman University.
The experiment uses a robot object named BUD that has three pillars, each pillar has five tiles and each tile represents a playing card in the game [SET](https://en.wikipedia.org/wiki/Set_(card_game)).
Participants are instructed to turn the pillars manually until they think there is a valid set facing them, computer vision is then used in order to check if the cards facing the user are indeed a valid set or not.
This code runs on a Raspberry Pi that is connected to a camera and connected to BUD via Wifi in order to give feedback to the user according to the result that was processed.

## How it works
The [workflow notebook](workflow.ipynb) includes a detailed look into how a frame is processed.
