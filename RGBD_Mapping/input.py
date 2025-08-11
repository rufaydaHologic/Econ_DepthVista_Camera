def get_integer(string, minimum, maximum, error_message='Invalid option', check_expr=None):
    '''
    Funciton Name: get_integer
    Description: This function is used to get a valid integer input from the user.
    :param string: The message for getting input
    :type string: str
    :param minimum: minimum valid value of the input
    :type minimum: int
    :param maximum: maximum valid value of the input
    :type maximum: int
    :param error_message: Error message which needs to be shown when input is not valid
    :type error_message: str
    :param check_expr: expression which needs to be checked, by default its none.
    :type check_expr: Any
    :return: valid integer from the user
    :rtype: int
    '''

    integer = -1
    while True:
        try:
            integer = int(input(string))
            if maximum >= integer >= minimum:
                if check_expr is None or check_expr(integer):
                    break
                continue
            else:
                print(error_message)
                continue
        except ValueError:
            print(error_message)
            continue

    return integer
