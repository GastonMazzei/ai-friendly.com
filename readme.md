# AI Playground

This is a simple Flask App which allows the user to upload a training set (`LearnCard.xlsx`) file to train a two
layer neural network. In addition the user uploads an `InCard.xlsx` spreadsheet with data to be run by the network.
Once the process is finished an email is sent to the user with the results XLSX from running the 
network to the `InCard`.


# Installation and usage

To use you need to have a Linux machine with `docker` and `docker-compose` installed and working. To start simply do:

```
$ docker-compose up -d
```

and point your browser to `http://localhost/`. In addition an instance of Jenkins is launched used to process the
requests in the background. You can access it at: `http://localhost:8080`.

# Notes

- If you add plugins to Jenkins, to reduce the size of the .git directory you should add only the `jenkins_home/plugins/<plugin-name>.jpi` file to the repo but add the corresponding directory to the `.gitignore` (`jenkins_home/plugins/<plugin-name>/`)

# References

 - [Container configuration for flask app](https://medium.com/bitcraft/docker-composing-a-python-3-flask-app-line-by-line-93b721105777)
