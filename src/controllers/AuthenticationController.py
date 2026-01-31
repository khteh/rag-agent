import logging, jsonpickle
from quart import flash, request, json, Blueprint, session, render_template, session, redirect, url_for
from marshmallow import ValidationError
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from datetime import datetime, timezone
from base64 import b64decode
from src.common.ResponseHelper import Respond
from src.models.UserModel import UserModel, UserSchema
from src.common.Authentication import Authentication
from src.common.Response import custom_response
from src.models import engine
auth_api = Blueprint("auth", __name__)
user_schema = UserSchema()
@auth_api.context_processor
def inject_now():
    return {'now': datetime.now(timezone.utc)}

@auth_api.route("/login", methods=["GET", "POST"])
async def login():
    """
    User Login
    """
    if request.method == "POST":
        try:
            form = await request.form
            req_data = {"email": form["username"], "password": form["password"]}
            with Session(engine) as dbsession:
                data = user_schema.load(req_data, partial=True, session=dbsession)
            if not data:
                await flash("Invalid input!", "danger")
                return await Respond("login.html", title="Welcome to Python Flask RESTful API", error=data["Invalid input!"])
            if not data.email or not data.password:
                await flash("You need an email and password to login", "danger")
                return await Respond("login.html", title="Welcome to Python Flask RESTful API", error="You need an email and password to login")
            user = UserModel.get_user_by_email(data.email)
            if not user:
                await flash("Invalid user!", "danger")
                logging.warning(f"[Auth] Invalid user {data.email}!")
                return await Respond("login.html", title="Welcome to Python Flask RESTful API", error="Invalid user!")
            if not user.check_hash(data.password):
                await flash("Invalid email or password!", "danger")
                logging.warning(f"[Auth] Invalid email or password {user.email}!")
                return await Respond("login.html", title="Welcome to Python Flask RESTful API", error="Invalid email or password!")
            user.token = Authentication.generate_token(user.id) # decoded_token: {'data': {'user_id': 1}, 'error': {}}
            session["user"] = {"id": user.id, "email": user.email, "token": user.token}
            logging.debug(f"login(): {session["user"]}")
            data.lastlogin = datetime.now(timezone.utc)
            user.update(data)
            """
            sqlalchemy.orm.exc.DetachedInstanceError: Instance <UserModel at 0x7af63c6e2250> is not bound to a Session; attribute refresh operation cannot proceed (Background on this error at: https://sqlalche.me/e/20/bhk3)
            https://docs.sqlalchemy.org/en/20/errors.html#error-bhk3
            https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html
            """
            #session['user'] = jsonpickle.encode(user)
            logging.info(f"[Auth] User {user.email} logged in")
            if "url" in session and session["url"]:
                return redirect(session["url"])
            else:
                return redirect(url_for("home.index"))
        except ValidationError as err:
            errors = err.messages
            valid_data = err.valid_data
            logging.exception(f"[Auth] login() exception! {errors}")
            return await Respond("login.html", title="Welcome to Python Flask RESTful API", error=errors)
    return await Respond("login.html", title="Welcome to Python Flask RESTful API")

#@auth_api.route("/login_oidc", methods=["GET", "POST"])
#@oidc.require_login
#async def login_oidc():
    """
    https://github.com/kroketio/quart-keycloak for OIDC
    Example for protected endpoint that extracts private information from the OpenID Connect id_token.
       Uses the accompanied access_token to access a backend service.
    """
#    if oidc.user_loggedin:
#        try:
#            info = oidc.user_getinfo(['preferred_username', 'email', 'sub'])
#            username = info.get('preferred_username')
#            email = info.get('email')
#            user_id = info.get('sub')
#            if user_id in oidc.credentials_store:
#                try:
#                    access_token = oidc.get_access_token()
#                    pre, tkn, post = access_token.split('.')
#                    tkn += "=" * ((4 - len(tkn) % 4) % 4)
#                    token = json.loads(b64decode(tkn))
                    #print(f"token: {token}")
#                    roles = token["realm_access"]["roles"]
#                    headers = {'Authorization': 'Bearer %s' % (access_token)}
#                    session["user"] = {"id": user_id, "username": username, "email": email, "token": access_token, "roles": roles}
                    # YOLO
                    #greeting = requests.get('http://localhost:8080/greeting', headers=headers).text
                    #return await Respond("index.html")
#                   print(f"[Auth] UserId: {user_id}, UserName: {username}, Email: {email}, Roles: {roles}, Profile: {profile} logged in")
#                   if "url" in session and session["url"]:
#                       return redirect(session["url"])
#                   else:
#                       return redirect(url_for("home.index"))
#               except Exception as e:
#                   print(f"Login fails! {str(e)}")
#                   return await Respond("login_error.html", error_message="Invalid credentials!")
#           else:
#               print(f"Invalid credentials!")
#               return await Respond("login_error.html", error_message="Invalid credentials!")
#       except Exception as e:
#            print(f"Please login to continue. {str(e)}")
#        return await Respond("login_error.html", error_message="Invalid credentials!")

@auth_api.route("/logout")
@Authentication.auth_required(None)
async def logout():
    """
    User Logout
    """
    logging.info(f"[Auth] User logged out")
    session["url"] = None
    session["user"] = None
    return redirect(url_for("home.index"))

#@auth_api.route('/logout_oidc')
#@oidc.require_login
#async def logout_oidc():
#    """Performs local logout by removing the session cookie."""
#    if oidc.user_loggedin and session["user"]:
#        print(f"[Auth] UserId: {session['user']['id']}, UserName: {session['user']['username']}, Email: {session['user']['email']} logged out")
#        oidc.logout()
#        session["user"] = None
#        session["url"] = None
#    return redirect(url_for("home.index"))

@auth_api.route("/profile")
@Authentication.auth_required("auth.profile")
async def profile():
    """
    Get my profile
    """
    u = session['user']
    user = UserModel.get_user(u.id)
    if not user:
        raise Exception(f"User {u.id} not found!")
    return custom_response(user_schema.dump(user), 200)