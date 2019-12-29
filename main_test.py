import main

def test_index():
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/')
    assert r.status_code == 200
    assert '[[19 22]\n [43 50]]' in r.data.decode('utf-8')
