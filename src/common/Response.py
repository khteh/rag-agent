from quart import Response, json
def custom_response(result, code):
    """
    Custom Response
    """
    return Response(mimetype="application/json", response=json.dumps(result), status=code)
