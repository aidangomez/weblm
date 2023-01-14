from weblm.controllers.basic.controller import Controller as BasicController
from weblm.controllers.command.controller import Controller as CommandController

registry = {"basic": BasicController, "command": CommandController}