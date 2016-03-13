# zbMATH Classification Service

A service for automated classification of mathematical documents w.r.t the MSC classification system.
This project consists of three parts:

  (1) The prepare_classification-script that generates (given a document corpus) all necessary data needed for classification
  
  (2) The classificaiton-module that calucates classifications based on the generated data from the prepare_classification-script
  
  (3) A webservice that provides an interface to the classification-module via POST request and a simple html frontend.

## Dependencies
In order to run the scripts in the project, you need to install [scikit-learn](http://scikit-learn.org/stable/install.html), [nltk](http://www.nltk.org/install.html) and [Flask](http://flask.pocoo.org/)

## Usage

### The corpus format
The first line of the corpus must be a header, describing the following documents followed by an arbitrary number of documents in json format. Each line must contain an enclosed json document.

#### Example header:
The header consista of a name (has no effect to the classification) and a list of attributes that define the following documents

    {"relation-name" : "some-name", "attributes" : [{"name" : "title", "type" : "string"}, {"name" : "abstract", "type" : "string"}]}
    
#### Example document:
Each document must contain a document identifier, a list of MSC classifications and a list of values that match to the provided attributes in the header.
    
    [["1234.56789",["37E30","37G20","37J10"]],["Astonishing Discoveries in Mathematics","Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum."]]
    
    
