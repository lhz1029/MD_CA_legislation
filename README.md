# MD_CA_legislation

A quick analysis on legislation proposed in previous sessions in Maryland and California. This data set (not uploaded here) contains 
14,769 bills and comes from a json file created and maintained by FiscalNote. side from minor adjustments, the data is as-is from 
the original source. 

##Installation

I submitted my project in a iPython Notebook (.ipynb) format, as well as in a corresponding .py format. For best viewing, you 
can see the entirety of my project via the iPython Notebook, including the output of the programmed cells and inline visualizations.

Requirements: To run the code on the iPython Notebook (the output from my running the code is already shown), you must first install 
Jupyter Notebook (http://jupyter.readthedocs.org/en/latest/install.html).

###Next Steps
With more time, I would like to explore both the EDA and the model building further. In particular, I would also like to:

1. Build more nuance into my model labels. Maybe instead of just pass or not, I could have four labels for passed in both houses, passed in the House, passed in the Senate, and failed in both.
2. Improve my model accuracy. This would include optimizing for my hyperparamters and experimenting with more classifiers. This could also end in an ensemble classifier of different types of classifiers. I would also add more features, some ideas of which I describe below.
3. Look for more temporal patterns, such as whether a greater proporation of bills tend to be passed right before election period, etc.
This could come in the form of some time series analysis, plots, and including features like the number of days the bill was in debate 
(i.e. from the day of introduction to the day of voting).
4. Incorporate insights from my latent dirichlet model into my predictive random forest model. For example, using the documents' topic distributions as features in the model building would be similar but possibly more interesting and interpretable than just using bag of words.
5. Spend more time parsing words. These would include words that are commonplace in legislation but not necessarily the English language.
6. Perform some analysis on the verbs used, since legislation has a lot of do with garnering action, and it could be informative to see which verbs tended to be in the most successfull bills and possibly even incorporate that in my models. I had originally intended to do such, as I parsed both the nouns and verbs from the documents, but ultimately just stuck with nouns for simplicity.
7. Incorporate more people features into the model. For example, this would include having dummy indicators for all the individuals and whether or not they sponsored or cosponsored the bill, etc. I would also like to perform more EDA on the representatives, such as seeing what sorts of bills different representatives tend to supprt/oppose and measure these differences between individuals to see who are most similar, etc.
