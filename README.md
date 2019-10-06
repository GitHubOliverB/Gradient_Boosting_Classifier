<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/GitHubOliverB/Gradient_Boosting_Classifier">
    <img src="GBT_Classifier_Example\Plots\Visualization\GBT_Classifier_Crosstraining_0_Visualization_Signal_Probabilty_XY_Background_No_Data.png" alt="Logo" width="504" height="420">
  </a>
  <h3 align="center">A Gradient-Boosting-Classifier Example In Python</h3>
  <p align="center">
    An awesome semi-automated classifier using Gradient-Boosting with lots of useful and informative plots!
    <br />
    <a href="https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Features](#features)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Preparation](#preparation)
  * [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

![alt text](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/GBT_Classifier_Example/Plots/Decision_Trees/GBT_Classifier_Example_0_Decision_Tree_0.png)

There are many Gradient-Boosting-Classifier templates available on GitHub, however, 
I didn't find one that needs heavy editing or was automated enough for an easy use.
Also most seem to lack important figures and plots, helping you to optimize your setup and deal with overfitting.
I want to create a template so amazing that it'll be the last one you ever need. 

Of course, no template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue.

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Built With
This project was build using Python 3.7.4 and the following Python libraries installed:

* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [matplotlib](http://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [pydotplus](https://pypi.org/project/pydotplus/)

<!-- FEATURES -->
## Features

* Classifying your data as signal/background using Gradient-Boosting-Decision-Trees.
* Grid Search for the average (cv=k) best performing hyperparameters.
* Easy access to hyperparameters and other settings.
* Separated correlation-matrix & box-plots for signal/background.
* Option to save all Decision-Trees figures.
* Signal-Probability-Plot for signal/background class and training & testing with a Kolmogorov-Smirnov test to check for under/overtraining.
* Feature-Importance(+STD), ROC-Curve and Precision-Recall-Curve plots.
* 1 - AUC-ROC/P-R vs. Number Of Trees plots.
* AUC-ROC vs. Training Sample Size for training & testing plot
* (Neat 2-dimensional visualization for two feature problems)
* Crosstraining (k=2).

<!-- GETTING STARTED -->
## Getting Started

This is an example on how to use this template for your own classification problems.
To get a local copy up and running follow these simple example steps.

### Installation

1. Install Python and all libraries needed.
2. Clone the repo.
```sh
git clone https://github.com/GitHubOliverB/Gradient_Boosting_Classifier.git
```
### Preparation

Before you can use this template, you need to setup your data and adjust your parameters. So let's start with the setup:

The data you want to use for your training and testing needs to be formatted in at least two .csv files.
You'll need a seperated file for all classes that you want to use. Split your data according to the signal-class 
(positive events, the one you are interested in) and the background-class(es). Furhtermore you don't need to include 
a label/onehotencoding column for your signal/background, as this will be added later on. Just the independent features are enough!

Temporarily, two columns ('Signal_Indicator', 'Background_Indicator') will be added to the dataframes which is generated from the .csv files.
The values (1,0) are assigned to each dataset from the [Signal directory](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/tree/master/Data/Signal).
The values (0,i) are assigned to the i-th dataset from the [Background directory](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/tree/master/Data/Background).

At the moment all background classes will simply be grouped as one background. 

## Usage

I) Put the .csv files in the corresponding subdirectory in [Data dir](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/tree/master/Data). 

All files you put in the Data/Signal and Data/Background dir will be added and used and are assumed to be csv files.

Skip:

I generated my own data by running the [Gaussian_Generator script](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/Gaussian_Generator.py). You can take a look at it and play around with it if you want.
I used 1 signal and 4 backgrounds, all following 2-dimensional gaussian distributions:
![alt text](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/Gaussian_Plot.png)


II) In the [Feature_File](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/Feature_File.py) is a list defined as Feature_List. Put all features (names as strings) you want to use for the classification there.

Optional: Go into [Grid Search](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/GridSearch.py) down to Block 3 and adjust the search parameters to your liking.
It will return the best set of hyperparameters for your classifier (from the ones you specified). 

WARNING: This can take long, so use with some thought behind it!

III) Head over to the [Parameters File](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/Parameters.py) to adjust the hyperparameters of your classifier, the name of the classifier(and dir) etc.

IV) Run the [Training](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/BDT_Training_Testing.py).

Optional: In the training script, Block 4.0, you can set the last argument of classifier_training to 'True' is you are interested in the Decision Trees.

Here is an example for the [Output](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/Output_Example.txt).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Oliver Bey - oliver.bey91@gmail.com



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/GitHubOliverB/Gradient_Boosting_Classifier.svg?style=flat-square
[contributors-url]: https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/GitHubOliverB/Gradient_Boosting_Classifier.svg?style=flat-square
[forks-url]: https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/network/members
[issues-shield]: https://img.shields.io/github/issues/GitHubOliverB/Gradient_Boosting_Classifier.svg?style=flat-square
[issues-url]: https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/issues
[license-shield]: https://img.shields.io/github/license/GitHubOliverB/Gradient_Boosting_Classifier.svg?style=flat-square
[license-url]: https://github.com/GitHubOliverB/Gradient_Boosting_Classifier/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/oliver-bey-2b148918b/
