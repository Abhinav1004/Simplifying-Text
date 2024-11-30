# SIMSUM: Document-level Text Simplification via Simultaneous Summarization

## Folder Structure
``` 
SimSum/
├── datasets/                   
│   ├── d_wiki/             # Directory for D-Wikipedia data
│   └── wiki_doc/            # Directory for WikiDoc data
├── docs/                   
│   ├── figures/             # Images for documentation
│   └── presentation_slides/ # Presentation slides
├── utils/                 
│   ├── baseline_models/         # Baseline models implementation       
│   ├── evaluate_model/         # Implementation for evaluation metrics
│   ├── processing/              # Data processing implementation
│   └── simsum_models/           # SimSum model implementation
├── main.py                 # Entry point for training
├── generate_plots.py        # Script to generate plots for training and evaluation metrics
├── train_valid_data_generator.py   
├── utils.py                
├── requirements.txt        # List of required packages
├── Simplifying_Text_Colab.ipynb             # Colab Training notebook 
├── NLP_Project_Presentation_Final.pptx             # Presentation slides
└── README.md               # Project documentation

```

# Setup Instructions

1. Unzip the file **cs23mtech15001_cs23mtech15020.zip** in a directory
2. Please create a virtual environment using the command `virtualenv venv` and use `source bin/activate` to activate the same
3. Install the required libraries using the command `pip install -r requirements.txt`
4. Install Easse Library
   ```
      1. git clone https://github.com/feralvam/easse.git
      2. pip install -e ./easse
   ```
4. To train and evaluate the model from terminal 
   1. Modify the parameters in the file `main.py` as per the requirement
   2. Run the command `python main.py` to train and evaluate the trained the model 
5. To generate the plots for the training and evaluation metrics, run the command `python generate_plots.py`
6. To train and evaluate the model on colab/Jupyter notebook 
   1. Launch the jupyter notebook by executing `jupyter notebook` in the current directory
   2. Execute all cells of the notebook `Simplifying_Text_Colab.ipynb.ipynb` to do the training in Jupyter notebook
7. Please find the final presentation slides in  `docs/presentation_slides/NLP_Project_Presentation_Final.pptx`


