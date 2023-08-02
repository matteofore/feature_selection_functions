# feature_selection_functions

Simple but yet effective functions for developing a statistically effective feature selection and engineering.

The functions provide an approach based on the Pearson correlation, the ANOVA table and the chi squared contingnecy table.
In particular, the driver_selection function assess automatically whether the target is continous or binary and accordingly to that, it applies the appropiate method.
If the target is continous:
- if the variable is continous, the Pearson correlation is applied and the variable is selected if the correlation value is above the threshold_r2 in absolute terms.
- if the variable is binary/categorical, the ANOVA is computed and the variable is selected if the p-value, associated to it, is lower than the one specified in the function as parameter (p_value). It's worth to mention that the categorical variable should be transformed into dummies through one-hot enconding or other similar techniques to allow this method to work.
If the target is binary:
- if the variable is continous, the ANOVA is applied with the same logic specified above.
- if the variable is binary/categorical, the chi2_table function is applied where its strictness can be regolated through the threshold_phi parameter. Consider that higher value of phi would lead to features being selected more easily. An appropriate value is 0.05.

Then the features selected are returned by the functions and the continous and categorical variable can be viewed through their corresponding variables.

Moreover, two extra functions that address the problem of multicollinearity among features are provided:
- identify_mult_features function identify the features that are correlated between them with respect to a selected threshold.
- remove_mult_features function removes features that are affected by multicollinearity.
