import re, logging, json, jsonpickle
from quart import request, Blueprint, session, render_template, flash, redirect, url_for
from marshmallow import ValidationError
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from datetime import datetime, timezone
from src.common.ResponseHelper import Respond
from src.models.UserModel import UserModel, UserSchema
from src.common.Authentication import Authentication
from src.common.Response import custom_response
from src.models import engine
user_api = Blueprint("user", __name__)
user_schema = UserSchema()
@user_api.context_processor
def inject_now():
    return {'now': datetime.now(timezone.utc)}

@user_api.get("/")
@user_api.get("/index")
async def index():
    """
    User Index page
    """
    return await Respond("users.html", title="Welcome to Python RESTful API", users=UserModel.get_users())
	
@user_api.route("/create", methods=["GET","POST"])
async def create():
    """
    Create User
    """
    if request.method == "POST":
        try:
            form = await request.form
            if not form["firstname"]:
                await flash("Please provide firstname!", "danger")
                return redirect(url_for("user.create"))
            if not form["lastname"]:
                await flash("Please provide lastname!", "danger")
                return redirect(url_for("user.create"))			   
            emailRegex = r"[\w.-]+@[\w.-]+.\w+"
            if not re.match(emailRegex, form["email"]):
                await flash("Please provide an valid email address!", "danger")
                return redirect(url_for("user.create"))
            if not form["password"]:
                await flash("Please provide an valid password!", "danger")
                return redirect(url_for("user.create"))
            if not form["password1"] or form["password"] != form["password1"]:
                await flash("Password mismatch!", "danger")
                return redirect(url_for("user.create"))
            if UserModel.isExistingUser(form['email']):
                await flash(f"Trying to register an existing user {form['email']}!", "danger")
                return redirect(url_for("user.create"))							
            req_data = {
               "firstname": form["firstname"],
			   "lastname": form["lastname"],
			   "email": form["email"],
               "password": form["password"]
			}
            logging.debug(f"create() request data: {req_data}")
            # Validate input data against UserModel
            with Session(engine) as dbsession:
                data = user_schema.load(req_data, session=dbsession)
                logging.debug(f"user.create(): {data.serialized}")
		    # Check if user already exists in the database
            if not data:
                await flash(f"Invalid input!", "danger")
                return redirect(url_for("user.create"))
            if UserModel.isExistingUser(data.email):
                await flash(f"Trying to create an existing user!", "danger")
                return redirect(url_for("user.create"))
            id = UserModel.add(data)
            user = UserModel.get_user(id)
            logging.debug(f"New user id: {user.id}, email: {user.email} created successfully!")
            token = Authentication.generate_token(user.id)
            session['user'] = jsonpickle.encode(user)
            logging.debug(f"session['user']: {session['user']}")
            #print(f"create() token: {token}")
            await flash(f"New user id: {user.id}, email: {user.email} created successfully!", "success")
            return redirect(url_for("user.index"))
        except ValidationError as err:
            valid_data = err.valid_data	
            logging.exception(f"create() exception! {err.messages}")
            await flash(f"Failed to create user! {err.messages}", "danger")
            return redirect(url_for("user.create"))
    return await Respond("user_create.html", title="Welcome to Python Flask RESTful API")

@user_api.route("/all")
@Authentication.auth_required("user.get_all")
async def get_all():
    return custom_response(user_schema.dump(UserModel.get_users(), many=True), 200)

@user_api.route("/<int:id>", methods=["GET"])
@Authentication.auth_required("user.get_user")
async def get_user(id):
    """
    Get a user
    """
    user = UserModel.get_user(id)
    if not user:
        logging.warning(f"User id: {id} not found!")
        return custom_response({"error": f"User {id} not found!"}, 400)
    return custom_response(user_schema.dump(user), 200)

@user_api.route("/update/<int:id>", methods=["PUT"])
@Authentication.auth_required("user.update")
async def update(id):
    """
    Update user 'id'
    """
    try:
        req_data = await request.get_json()
        data = user_schema.load(req_data, partial=True)
        if not data:
            await flash(f"Failed to update user {id} with invalid input data!", "warning")
            return redirect(url_for("user.index"))
        user = UserModel.get_user(id)
        if not user:
            await flash(f"Trying to update non-existing user {id}!", "warning")
            return redirect(url_for("user.index"))
        user.update(data)
        session['user'] = jsonpickle.encode(user)
        logging.info(f"User {user['email']} updated user {id} successfully!")
        await flash(f"User {id} updated successfully!", "success")
    except ValidationError as err:
        errors = err.messages
        valid_data = err.valid_data
        logging.exception(f"Failed to update user {id}! Exception: {errors}")
        await flash(f"Failed to update user {id}! Exception: {errors}!", "danger")
    return redirect(url_for("user.index"))

@user_api.route("/delete/<int:id>", methods=["DELETE"])
@Authentication.auth_required("user.delete")
async def delete(id):
    """
    Delete user 'id'
    """
    try:
        user = UserModel.get_user(id)
        if not user:
            await flash(f"Trying to delete non-existing user {id}!", "warning")
            return redirect(url_for("user.index"))
        user.delete()
        session['user'] = None
        logging.info(f"User {user['email']} deleted user {id} successfully!")
        await flash(f"User {id} deleted successfully!", "success")
    except ValidationError as err:
        errors = err.messages
        valid_data = err.valid_data	
        logging.exception(f"User {user['email']} failed to delete user {id}! Exception: {errors}")
        await flash(f"Failed to delete user {id}! Exception: {errors}", "danger")
    return redirect(url_for("user.index"))