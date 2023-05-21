import json
import network
import socket
from time import sleep

ssid = ''
password = ''

def connect():
    # Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        sleep(1)
    ip = wlan.ifconfig()[0]
    print(f'Connected on {ip}')
    return ip


def open_socket(ip):
    # Open a socket
    address = (ip, 80)
    connection = socket.socket()
    connection.bind(address)
    connection.listen(1)
    return connection


def recvall(sock):
    data = ''
    count = 0
    while True:
        part = sock.recv(BUFFER_SIZE)
        part = part.decode('utf-8')
        if count == 0 and part[0] != '{':
            return ''
        try:
            data += part
        except MemoryError as e:
            print(f"{len(data)=}")
            print(f"{len(part)=}")
            print(data)
            raise(e)

        if part.endswith(']]}'):
            # either 0 or end of data
            break
        count += 1

    if data.startswith('{"y": [') and data.endswith(']]}'):
        return data
    else:
        return ''

def webpage(charts):
    html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Equipment problem detection</title>
                <h1>Problem machines</h1>
                <p>These 4 charts show the most likely machines to have a problem with them with the % confidence that the machine is on with the problem</p>
                <style>
                    .graph-container {
                        width: 48%;
                        height: 300px;
                        margin: 10px;
                        display: inline-block;
                        box-sizing: border-box;
                    }
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id="graphs"></div>
                <script>
                    // Define the dictionary with titles and data
                    var graphData = {
                        {1},
                        {2},
                        {3},
                        {4}
                    };
                    var graphsContainer = document.getElementById('graphs');
                    Object.entries(graphData).forEach(function([title, data]) {
                        var graphDiv = document.createElement('div');
                        graphDiv.className = 'graph-container';
                        graphDiv.id = 'graph-' + title;
                        graphsContainer.appendChild(graphDiv);
                        var trace = {
                            x: Array.from({
                                length: data.length
                            }, (_, i) => i),
                            y: data,
                            mode: 'lines'
                        };
                        var layout = {
                            title: title,
                            margin: {
                                t: 30
                            },
                            xaxis: {
                                title: 'X-axis'
                            },
                            yaxis: {
                                title: 'Y-axis'
                            }
                        };
                        Plotly.newPlot(graphDiv, [trace], layout, {
                            displayModeBar: false
                        });
                    });
                </script>
            </body>
        </html>
    """
    for i, x in enumerate(charts[:4]):
        html = html.replace("{" + str(i + 1) + "}", x)
    return str(html)

ip = connect()
connection = open_socket(ip)

HOST = '192.168.0.246'
PORT = 9090
BUFFER_SIZE = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("Connecting to socket")
s.connect((HOST, PORT))
print("Connection made")


while True:
    data = recvall(s)
    if len(data) == 0:
        s.send("0".encode("UTF-8"))
        continue

    s.send("1".encode("UTF-8"))

    data = json.loads(data)

    # sorting data
    y = data['y']
    data = list(zip(*data['data']))
    cols = [
        'motor1', 'motor11', 'motor12', 'motor2', 'motor3', 'motor4',
        'motor5_6', 'motor7_8', 'pressure0', 'pressure1', 'pressure2',
        'pressure3', 'pressure4', 'pressure5', 'pressure6', 'motor5',
        'motor6', 'motor7', 'motor8'
    ]

    out = {}

    for i, elem in enumerate(zip(cols, y, data)):
        title = f"{elem[0]} | {round(elem[1]*100,2)}% problem sensor"
        out[title] = list(elem[2])

    out = list(
        sorted(
            out.items(),
            key=lambda x: float(x[0].split("| ")[-1].split("%")[0]),
            reverse=True))

    charts = []

    for elem in out:
        elem = repr({elem[0]: elem[1]})
        elem = str(elem.replace("'", '"')[1:-1])
        charts.append(elem)

    client = connection.accept()[0]
    request = client.recv(1024)
    html = webpage(charts)
    client.send(html)
    client.close()
    print("END WEB SERVE")
