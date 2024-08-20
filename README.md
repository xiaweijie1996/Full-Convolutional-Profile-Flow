# Full-Convolutional-Profile-Flow

<p align="center">
  <img src="materials/fcpflow.png" alt="Top Bar">
</p>

Welcome to the repository containing the implementation of algorithms from the paper titled **'A Flow-Based Model for Conditional and Probabilistic Electricity Consumption Profile Generation and Prediction'**.

- We have provided detailed tutorials for you to undershtand our model, check below.
- All process data for this project are available at [Data](data).

## Tutorials

To get started with our models and understand how they work, we have prepared several tutorials. Please click on the links below to explore them:

[Conditional Generation Tutorial](tutorial_conditioanl_gen.ipynb): (**Beginner friendly**) This tutorial guides you through generating electricity consumption profiles conditionally.

[Unconditional Generation Tutorial](tutorial_uncond_gen.ipynb): Learn how to generate electricity consumption profiles without specific conditions.

[Prediction Tutorial](tutorial_prediction.ipynb): Dive into predicting future electricity consumption profiles based on past data.

We highly recommend going through these tutorials in the order listed above to gain a comprehensive understanding of the models and their applications. Especially **Conditional Generation** tutorial.

## Code of other models
For the code of t-Copula used in the paper, please check: [t-Copula](https://github.com/MauricioSalazare/multi-copula)

For the of GAN, VAE, etc, please check: [Generative models](https://github.com/xiaweijie1996/Generative-Models-for-Customer-Profile-Generation).

## FCPFlow

![ffctfflow+str](https://github.com/xiaweijie1996/Full-Convolutional-Time-Series-Flow/assets/84010474/f29e1a10-0ae9-4a76-b20a-c9c1e5d781c3)

## Data Sources

The research uses raw data from the following open-source databases:

- **Netherlands Smart Meter Data**: [Liander Open Data](https://www.liander.nl/partners/datadiensten/open-data/data)
- **UK Smart Meter Data**: [London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- **Germany Smart Meter Data**: [Open Power System Data](https://data.open-power-system-data.org/household_data/2020-04-15)
- **AUS Smart Meter Data**: [Smart-Grid Smart-City Customer Trial Data](https://data.gov.au/dataset/ds-dga-4e21dea3-9b87-4610-94c7-15a8a77907ef/details)
- **USA Smart Meter Data**:  [DATAPORT](https://dataport.pecanstreet.org/)


## Contact and Citations
For inquiries, suggestions, or potential collaborations, please reach out to Weijie Xia at [w.xia@tudelft.nl](mailto:w.xia@tudelft.nl).

To cite the research paper related to this project, please use the following Bibtex entry:

Xia, W., Wang, C., Palensky, P., & Vergara, P. P. (2024). _A Flow-Based Model for Conditional and Probabilistic Electricity Consumption Profile Generation and Prediction_. arXiv preprint arXiv:2405.02180.

```bibtex
@article{xia2024flow,
  title={A Flow-Based Model for Conditional and Probabilistic Electricity Consumption Profile Generation and Prediction},
  author={Xia, Weijie and Wang, Chenguang and Palensky, Peter and Vergara, Pedro P},
  journal={arXiv preprint arXiv:2405.02180},
  year={2024}
}
