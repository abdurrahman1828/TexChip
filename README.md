# TexChip

**Interpretable Wood Chip Moisture Content Prediction through Texture Analysis**


## Files & Directories

- ```main.py``` can be directly used to generate results.
- ``utils.py`` includes the required utility functions for running other codes.
- The ``RF_with_LIME.py`` file contains the code to generate explanations of the predictions generated by the Random Forest model using LIME explainer.
- The `results``` folder will contain the saved results. It also includes codes to summarize results and generate plots.

- The ```data``` folder contains pre-extracted Haralick features from two datasets. This study 
was conducted on two datasets using k-fold cross-validation. Therefore, train means all ``k-1`` folds, and test means the remaining one fold. 


## License

This project is licensed under the [License](LICENSE).


## Contact

If you have any questions or suggestions, feel free to contact us:

- Email: ar2806@msstate.edu
- GitHub Issues: [Open an issue](https://github.com/abdurrahman1828/TexChip/issues)

## Acknowledgments

- Thanks to [mahotas library](https://github.com/luispedro/mahotas) for the code for extracting Haralick features.
- This project is funded by USDA-NIFA (Grant No. 2022-67022-37861 and 2020-67019-30772).

