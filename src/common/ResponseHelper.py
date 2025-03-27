from typing import Optional
from werkzeug.http import COOP
from secrets import token_urlsafe
from typing import Optional, cast
from quart import (
    Blueprint,
    Response,
    ResponseReturnValue,
    current_app,
    make_response,
    render_template,
    session
)

def _apply_security_headers(
    response: Response, nonce: Optional[str] = None
) -> Response:
    response.content_security_policy.default_src = "'self'"
    response.content_security_policy.base_uri = "'self'"
    response.content_security_policy.connect_src = (
        "'self' https://cloudflareinsights.com"
    )
    response.content_security_policy.form_action = "'self'"
    response.content_security_policy.frame_ancestors = "'none'"
    response.content_security_policy.frame_src = "https://www.youtube-nocookie.com"
    response.content_security_policy.img_src = "'self' data:"
    if nonce is not None:
        response.content_security_policy.script_src = (
            f"'self' 'nonce-{nonce}' https://static.cloudflareinsights.com"
        )
        response.content_security_policy.style_src = f"'self' 'nonce-{nonce}'"
    response.cross_origin_opener_policy = COOP.SAME_ORIGIN
    response.headers["Referrer-Policy"] = "no-referrer, strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

async def Respond(*args, **kwargs):
    nonce = token_urlsafe(12)
    body = await render_template(*args, **kwargs)
    body = body.replace('type="module"', f'type="module" nonce="{nonce}"')
    response = cast(Response, await make_response(body))
    return _apply_security_headers(response, nonce)
