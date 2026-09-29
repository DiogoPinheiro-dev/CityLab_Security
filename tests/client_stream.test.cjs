const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
    const callbacks = [];
    const element = () => ({ getContext: () => ({ drawImage() {} }),
        classList: { remove() {}, add() {} }, toBlob: cb => callbacks.push(cb), readyState: 2 });
    let now = 0;
    class Socket {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 1;
        send() {}
        close() {}
    }
    const context = vm.createContext({ document: { getElementById: element, createElement: element },
        window: { location: { protocol: 'http:', hostname: 'localhost' } },
        WebSocket: Socket, performance: { now: () => now }, console, setTimeout, clearTimeout });
    vm.runInContext(fs.readFileSync('Client/stream.js', 'utf8').replace(/bootstrap\(\);\s*$/, ''), context);
    const run = code => vm.runInContext(code, context);
    run('state.cameraReady = true; conectarWebSocket();');
    return { run, callbacks, time: value => { now = value; } };
}

const reply = payload => `state.ws.onmessage({data: ${JSON.stringify(JSON.stringify(payload))}})`;

test('two pending frames retain independent send times', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(300); s.run(reply({ frame: 1 }));
    assert.equal(s.run('state.lastRoundTripMs'), 200);
    s.time(400); s.run(reply({ frame: 2 }));
    assert.equal(s.run('state.lastRoundTripMs'), 200);
    assert.equal(s.run('state.inFlightFrames'), 0);
});

test('encoding failure preserves other pending timestamps', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()(null);
    assert.equal(s.run('state.pendingFrames.size'), 1);
    s.time(300); s.run(reply({ frame: 1 }));
    assert.equal(s.run('state.lastRoundTripMs'), 200);
});

test('old encoding callback cannot change reconnected session', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()');
    const old = s.callbacks.shift();
    s.run('conectarWebSocket()');
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    old({});
    assert.equal(s.run('state.inFlightFrames'), 1);
    // A nova conexao numera os frames de novo a partir de 1, como o servidor.
    assert.equal(s.run('state.pendingFrames.size === 1 && state.pendingFrames.has(1)'), true);
});

test('response is matched by its frame number, not by arrival order', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(500); s.run(reply({ frame: 2 }));
    assert.equal(s.run('state.lastRoundTripMs'), 300);
    assert.equal(s.run('state.pendingFrames.size === 1 && state.pendingFrames.has(1)'), true);
});

test('frame without response for 30 s closes the socket and schedules a reconnect', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(30099); s.run('checkFrameTimeout()');
    assert.equal(s.run('state.ws !== null'), true);
    s.time(30101); s.run('checkFrameTimeout()');
    assert.equal(s.run('state.ws === null && state.reconnectTimer !== null'), true);
    assert.equal(s.run('state.pendingFrames.size + state.inFlightFrames'), 0);
    s.run('clearTimeout(state.reconnectTimer)');
});

test('pausing clears the last results from the overlay', () => {
    const s = setup();
    s.run(reply({ frame: 1, gestos: [{ bbox: [1, 2, 3, 4], alerts: ['Rendicao'] }] }));
    assert.equal(s.run('state.results.gestos.length'), 1);
    s.run('handleToggleStream()');
    assert.equal(s.run('state.results.gestos.length'), 0);
});
