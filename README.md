[![License](https://img.shields.io/github/license/Lambda-School-Labs/allay-ds)](https://github.com/Lambda-School-Labs/allay-ds/blob/master/LICENSE)
![Python 3.7.6](https://img.shields.io/badge/python-3.7.6-blue)
![FastAPI](https://img.shields.io/github/pipenv/locked/dependency-version/Lambda-School-Labs/allay-ds/fastapi)
![Tensorflow](https://img.shields.io/github/pipenv/locked/dependency-version/Lambda-School-Labs/allay-ds/tensorflow)

[![Maintainability](https://api.codeclimate.com/v1/badges/0cb57994085c5522e552/maintainability)](https://codeclimate.com/github/Lambda-School-Labs/allay-ds/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/0cb57994085c5522e552/test_coverage)](https://codeclimate.com/github/Lambda-School-Labs/allay-ds/test_coverage)

🚫 The numbers 1️⃣ through 5️⃣ next to each item represent the week that part of the docs needs to be comepleted by.  Make sure to delete the numbers by the end of Labs.

# Allay

The Allay project is [deployed on Heroku](https://labs21-allay-fe.herokuapp.com/).

The data science content moderation API is [deployed on Heroku](https://allay23-staging-ds.herokuapp.com/docs).

## 5️⃣ Contributors

🚫Add contributor info below, make sure add images and edit the social links for each member. Add to or delete these place-holders as needed

|[Alex Jenkins-Neary](http://www.alexjenkinsneary.com)|[Caleb Spraul](https://jcs-lambda.github.io)|[Andrew Archie](https://baiganking.github.io)|
| :----: | :----: | :----: |
|[<img src="https://www.dalesjewelers.com/wp-content/uploads/2018/10/placeholder-silhouette-male.png" width = "200" />](https://github.com/alexmjn)|[<img src="https://www.dalesjewelers.com/wp-content/uploads/2018/10/placeholder-silhouette-male.png" width = "200" />](https://github.com/jcs-lambda)|[<img src="https://www.dalesjewelers.com/wp-content/uploads/2018/10/placeholder-silhouette-male.png" width = "200" />](https://github.com/BaiganKing)|
|[<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/alexmjn)[<img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15">](https://www.linkedin.com/in/alexjenkinsneary)|[<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/jcs-lambda)[<img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15">](https://www.linkedin.com/)|[<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/BaiganKing)[<img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15">](https://www.linkedin.com/in/andrew-archie-04b24b1a9)|


🚫 5️⃣ Optional examples of using images with links for your tech stack, make sure to change these to fit your project

![Typescript](https://img.shields.io/npm/types/typescript.svg?style=flat)
[![Netlify Status](https://api.netlify.com/api/v1/badges/b5c4db1c-b10d-42c3-b157-3746edd9e81d/deploy-status)](netlify link goes in these parenthesis)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

🚫 more info on using badges [here](https://github.com/badges/shields)

## Project Overview

[Trello Board](https://trello.com/b/wDXK4crl/allay23)

[Product Canvas](https://www.notion.so/Allay-eb3c5b88ffab4ff199663cb40fcc1402)

[Deployed Project](https://labs21-allay-fe.herokuapp.com/)

[Deployed DS API](https://allay23-staging-ds.herokuapp.com/docs)

### Tech Stack

We use Python in Jupyter Notebooks to [explore and model the data](./exploration).
We then save that model and implement it within a [FastAPI app](./allay-ds-api),
which is [deployed to Heroku](https://allay23-staging-ds.herokuapp.com/docs) 
for live classification of Allay user generated content.

We use [Weights & Biases](https://www.wandb.com) for machine learning tracking
and to automate and report hyperparameter tuning for model optimization.

We use [Keras with Tensorflow](https://www.tensorflow.org/guide/keras/overview)
for modeling.

### Predictions

We implement Natural Language Processing using a text classifier model to
categorize whether reviews posted to the Allay website are appropriate.

<img src="https://i.imgur.com/ccUMIze.png" width="500">

<img src="https://i.imgur.com/BW4K5W9.png" width="500">

### Explanatory Variables

Ultimately, the explanatory variable is the text that is posted to the website.
This gets broken down into numerical features which are then modeled.

### Data Sources
🚫  Add to or delete souce links as needed for your project

- [Hate and Abusive Speech on Twitter](https://github.com/ENCASEH2020/hatespeech-twitter) (code) -
[Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior](https://arxiv.org/abs/1802.00393) (paper)
- [Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language) (code) -
[Automated Hate Speech Detection and the Problem of Offensive Language](https://arxiv.org/abs/1703.04009) (paper)
- [Twitter Hate Speech](https://www.kaggle.com/vkrahul/twitter-hate-speech)
- [Allay user content](https://labs21-allay-fe.herokuapp.com/)

### Python Notebooks

🚫  Add to or delete python notebook links as needed for your project

[Data Exploration](./exploration/explore_data.ipynb)

[Baseline ML models](./exploration/train_ml_models.ipynb)

[Baseline neural network models](./exploration/train_nn_models.ipynb)

### 3️⃣ How to connect to the web API

[Allay frontend](https://github.com/Lambda-School-Labs/allay-fe)

[Allay backend](https://github.com/Lambda-School-Labs/allay-be)

### 3️⃣ How to connect to the data API

🚫 List directions on how to connect to the API here

[Allay DS API Documentation](https://allay23-staging-ds.herokuapp.com/docs)

[Allay DS API Redoc Documentation](https://allay23-staging-ds.herokuapp.com/redoc)

## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a [code of conduct](./code_of_conduct.md). Please follow it in all your interactions with the project.

### Issue/Bug Request

 **If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**
 - Check first to see if your issue has already been reported.
 - Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
 - Create a live example of the problem.
 - Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes,  where you believe the issue is originating from, and any potential solutions you have considered.

### Feature Requests

We would love to hear from you about new features which would improve this app and further the aims of our project. Please provide as much detail and information as possible to show us why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this app, please submit a pull request. It is best to communicate your ideas with the developers first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.
- You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).

## Documentation

See [Allay backend](https://github.com/Lambda-School-Labs/allay-be/blob/master/README.md) for details on the backend of our project.

See [Allay frontend](https://github.com/Lambda-School-Labs/allay-fe/blob/master/README.md) for details on the front end of our project.

See [API Documentation](./allay-ds-api/README.md) for details on the data science API of our project.

See [Exploration readme](./exploration/README.md) for details on Weights and Biases hyperparameter sweeps.
