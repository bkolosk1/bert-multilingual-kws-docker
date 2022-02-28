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


Developed by:
Boshko Koloski @ IJS 