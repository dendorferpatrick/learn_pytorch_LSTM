from sacred import Experiment

ex = Experiment('hello_config')

@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient
    testvariable="hello"
@ex.automain
def my_main(message):
    print(message)
