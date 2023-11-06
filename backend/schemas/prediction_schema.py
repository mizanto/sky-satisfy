from marshmallow import Schema, fields, validate


class PredictionSchema(Schema):
    """
    Schema for validating prediction API input data.

    Fields:
    - customer_type: Type of the customer, either 'loyal_customer' or
                     'disloyal_customer'.
    - age: Age of the customer, must be between 0 and 120.
    - type_of_travel: Type of travel, either 'business_travel' or
                      'personal_travel'.
    - flight_distance: Distance of the flight in miles, required field.
    - ease_of_online_booking: Ease of online booking, rated between 0 and 5.
    - online_boarding: Online boarding experience, rated between 0 and 5.
    - class_: Class of the flight, either 'business', 'eco', or 'eco_plus'.

    The schema uses the Marshmallow library for validation.
    """
    customer_type = fields.Str(required=True,
                               validate=validate.OneOf(['loyal_customer',
                                                        'disloyal_customer']))
    age = fields.Int(required=True, validate=validate.Range(min=0, max=120))
    type_of_travel = fields.Str(required=True,
                                validate=validate.OneOf(['business_travel',
                                                         'personal_travel']))
    flight_distance = fields.Int(required=True)
    ease_of_online_booking = fields.Int(required=True,
                                        validate=validate.Range(min=0, max=5))
    online_boarding = fields.Int(required=True,
                                 validate=validate.Range(min=0, max=5))
    class_ = fields.Str(required=True,
                        data_key='class',
                        validate=validate.OneOf(
                            ['business', 'eco', 'eco_plus']))
