from flask.cli import FlaskGroup


from project import app
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

cli = FlaskGroup(app)


if __name__ == "__main__":
    cli()
