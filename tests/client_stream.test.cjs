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
    vm.runInContext(fs.readFileSync('Client/index.js', 'utf8').replace(/bootstrap\(\);\s*$/, ''), context);
    const run = code => vm.runInContext(code, context);
    run('state.cameraReady = true; conectarWebSocket();');
    return { run, callbacks, time: value => { now = value; } };
}

test('two pending frames retain independent send times', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(300); s.run('state.ws.onmessage({data: "{}"})');
    assert.equal(s.run('state.lastRoundTripMs'), 200);
    s.time(400); s.run('state.ws.onmessage({data: "{}"})');
    assert.equal(s.run('state.lastRoundTripMs'), 200);
    assert.equal(s.run('state.inFlightFrames'), 0);
});

test('encoding failure preserves other pending timestamps', () => {
    const s = setup();
    s.time(100); s.run('sendFrameToBackend()'); s.callbacks.shift()({});
    s.time(200); s.run('sendFrameToBackend()'); s.callbacks.shift()(null);
    assert.equal(s.run('state.pendingSentAt.length'), 1);
    s.time(300); s.run('state.ws.onmessage({data: "{}"})');
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
    assert.equal(s.run('state.pendingSentAt.length'), 1);
});
