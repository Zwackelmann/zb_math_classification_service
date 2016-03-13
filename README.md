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
For the sake of readability the json documents in the following examples are displayed in multiple lines, but in the corpus they must be written in one line.

#### Example header:
The header consista of a name (has no effect to the classification) and a list of attributes that define the following documents

    {
      "relation-name" : "some-name", 
      "attributes" : [
        {"name" : "title", "type" : "string"}, 
        {"name" : "abstract", "type" : "string"}
      ]
    }
    
#### Example document:
Each document must contain a document identifier, a list of MSC classifications and a list of values that match to the provided attributes in the header.
    
    [
      [
        "1234.56789",
        ["37E30","37G20","37J10"]
      ],
      [
        "Astonishing Discoveries in Mathematics",
        "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum."
      ]
    ]
    
    
## Preparation of Classification
To prepare the classification, execute the `prepare_classification.py` in the `src` folder. Deliver an output folder where the generated data shall be stored with the `-o` option followd by the document corpus.

    python src/prepare_classification.py -o output-folder data/zb_math-corpus.json
    
## Configuration of the Classification Service
Configure the input folder of the classification service in the `conf/application.conf`. This is the folder you created in the previous step.

    {"input-dir": "output-folder"}
    
## Start the Classification Service

To start the classification service, run the `start_service.py` script

    python src/start_service.py
    
The startup might take some time. When the service is loaded, it will print

    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    
You can send documents to the route `POST /classify` in order to receive classifications for that document.

The format must be

    { "title": "the title",
      "abstract": "the abstract" }
      
The Route `/` returns an example HTML frontend for classification
