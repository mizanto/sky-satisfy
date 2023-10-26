
def generate_field_info(field_name, field_rules):
    return f'<li><b>{field_name}</b>: {field_rules}</li>'


def generate_endpoint_info(title, method, description, params, example):
    param_list = '<ul>'
    for param, rules in params.items():
        param_list += generate_field_info(param, rules)
    param_list += '</ul>'

    return f'''
    <li>
        <b>{title}</b>: {method} request. {description}
        <ul>
            <li><b>Parameters:</b> {param_list}</li>
            <li><b>Example:</b> <code>{example}</code></li>
        </ul>
    </li>
    '''


def generate_api_endpoints_info():
    predict_params = {
        'customer_type': 'String - ["loyal_customer", "disloyal_customer"]',
        'age': 'Integer, range [0, 120]',
        'type_of_travel': 'String, - ["business_travel", "personal_travel"]',
        'flight_distance': 'Integer, required',
        'ease_of_online_booking': 'Integer, range [0, 5]',
        'online_boarding': 'Integer, range [0, 5]',
        'class': 'String, - ["business", "eco", "eco_plus"]'
    }

    predict_info = generate_endpoint_info(
        "/predict",
        "POST",
        "Make a prediction based on input data.",
        predict_params,
        '{"customer_type": "loyal_customer", "age": 25, ...}'
    )

    model_info = generate_endpoint_info(
        "/model/info",
        "GET",
        "Retrieve model information and metrics.",
        {},
        "None"
    )

    health_info = generate_endpoint_info(
        "/health",
        "GET",
        "Check the health status of the API.",
        {},
        "None"
    )

    return f'''
    <h1>API Endpoints</h1>
    <ul>
        {predict_info}
        <p>
        {model_info}
        <p>
        {health_info}
    </ul>
    '''
