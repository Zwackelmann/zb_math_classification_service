# zbMATH Classification Service

A service for automated classification of mathematical documents w.r.t the MSC classification system.
This project consists of three parts:

  (1) The prepare_classification-script that generates (given a document corpus) all necessary data needed for classification
  
  (2) The classificaiton-module that calucates classifications based on the generated data from the prepare_classification-script
  
  (3) A webservice that provides an interface to the classification-module via POST request and a simple html frontend.

## Dependencies

