# TheBrain - Matching facebook pages about to News Categories
A simple Flask application that can serve predictions from a scikit-learn model. Reads a pickled sklearn model into memory when the Flask app is started and returns predictions through the /predict endpoint. You can also use the /train endpoint to train/retrain the model. Any sklearn model can be used for prediction.

### Dependencies
- scikit-learn
- Flask
- pandas
- numpy

```
pip install -r requirements.txt
```

# Endpoints
### /predict (POST)
Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
{"pages":["about1", "about2","about3"]}
```

and sample output:
```
{
  "business": 1.6483516483516483,
  "entertainment": 1.8131868131868132,
  "politics": 2.6373626373626373,
  "sports": 2.1703296703296706,
  "tech": 5
}
```


### /Credits
https://github.com/amirziai/sklearnflask for Flask and Sklearn
