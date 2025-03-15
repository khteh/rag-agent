import os
from quart import Request, Response
# https://quart.palletsprojects.com/en/latest/how_to_guides/middleware.html
class AuthHeaderMiddleware:
    _auth_secret = os.getenv("AUTH_SECRET")
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if "headers" not in scope:
            return await self.app(scope, receive, send)
        if self._auth_secret:
            auth_header = scope["headers"].get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return Response(status_code=401, content="Missing or invalid token")
            if auth_header[7:] != self._auth_secret:
                return Response(status_code=401, content="Invalid token")
        return await self.app(scope, receive, send)

    async def error_response(self, receive, send):
        await send({
            'type': 'http.response.start',
            'status': 401,
            'headers': [(b'content-length', b'0')],
        })
        await send({
            'type': 'http.response.body',
            'body': b'',
            'more_body': False,
        })