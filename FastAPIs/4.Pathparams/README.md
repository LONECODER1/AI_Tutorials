The Path() function in fastAPI is used to provide metadata, validation rules,and documentation hints for path parameters in your API.

Title 
Description 
Ex:
ge,gt,le,lt,
Min_length
Max_length
regex

-------------------------------------------------

HTTP Exception is a special built-in exception in FastAPI used to return custom HTTP error responses when something goes wrong in your API.

Instead of returning a normal JSON ir crashing the server you cna gracefully raise an error with 

- a proper HTTP status codes(like 404,400,403,etc)
- a custom error message 
- extra headers
-------------------------------------------------

Query Parameter

it is an optional key-value pairs appended to the end of a URL used to pass additional data to the HTTP request.
They are used for filtering,sorting,searching and pagination without altering the endpoint path itself.
- ? => start of query params
- each is a key-value pair
- Multiple params are seperated by &

 ------------------------------------------------- 