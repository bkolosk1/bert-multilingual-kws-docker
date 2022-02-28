## Flask REST API for Multilingual keyword identification


This repository contains a docker for Multilingual keyword extraction using [BERT]

### Requirements
-  docker
-  docker-compose


#### Development

The following command

```sh
$ docker-compose up -d --build
```

will build the images and run the containers. If you go to [http://localhost:5000](http://localhost:5000) you will see a web interface where you can check and test your REST API.

#### Production

The following command

```sh
$ docker-compose -f docker-compose.prod.yml up -d --build
```

will build the images and run the containers. The web interface is now available through nginx server at [http://localhost](http://localhost).

#### Model training and citation
The model training is explained in the following work: 

```
@misc{koloski2022air,
      title={Out of Thin Air: Is Zero-Shot Cross-Lingual Keyword Detection Better Than Unsupervised?}, 
      author={Boshko Koloski and Senja Pollak and Blaž Škrlj and Matej Martinc},
      year={2022},
      eprint={2202.06650},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


Developed by:
Boshko Koloski @ IJS 