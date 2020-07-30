# Characterization and Classification of Bots using Reddit’s Comment Network

## About

This repository refers to a study to characterize and identify bots in a social network using only the network structure. This work considers a directed network of users constructed from comments in Reddit. The network characterization highlights the significant structural differences of bots, allowing them to be classified using only network features.

This work was guided by Professor [Daniel R. Figueiredo](http://www.land.ufrj.br/~daniel/) (PESC / COPPE / UFRJ).

## Publications

- [Full Paper on Brazilian Computing Society (pt-BR)](https://sol.sbc.org.br/index.php/wperformance/article/view/6471)  
Presented at the XXXIX Congresso da Sociedade Brasileira de Computação

- [Undergraduate Project (Poli / COPPE / UFRJ) (pt-BR)](http://monografias.poli.ufrj.br/monografias/monopoli10028221.pdf)  
as a partial fulfillment of the requirements for the degree of Computer and Information Engineer

## Using this repository

The code elaborated for this work is compatible with [Python 3.7.8](https://www.python.org/).

To install all dependencies and use Jupyter Notebook with a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
ipython kernel install --user --name=venv
jupyter notebook
```

In the `notebooks` folder are the [Jupyter Notebook](https://jupyter.org/) files that can be used to evaluate other datasets.

Reddit Comments datasets can be found in [pushshift](https://files.pushshift.io/reddit/comments/), the files should be placed uncompressed in `data/raw/`.
