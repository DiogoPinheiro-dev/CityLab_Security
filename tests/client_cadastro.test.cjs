const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
    const elements = {};
    const make = () => ({
        value: '', files: [], textContent: '', hidden: false, children: [],
        classList: { add() {}, remove() {} },
        addEventListener(type, handler) { this[`on${type}`] = handler; },
        setAttribute() {}, removeAttribute() {}, focus() {}, reset() {},
        append(...items) { this.children.push(...items); },
        replaceChildren(...items) { this.children = items; },
    });
    const element = id => (elements[id] ??= make());
    const revoked = [];
    const requests = [];
    const context = vm.createContext({
        document: { getElementById: element, createElement: make },
        window: { location: { protocol: 'http:', hostname: 'localhost', port: '' } },
        URL: { createObjectURL: file => `blob:${file.name}`, revokeObjectURL: url => revoked.push(url) },
        FormData,
        fetch: async (url, options) => {
            requests.push([url, options]);
            return { ok: true, json: async () => ({ mensagem: 'ok' }) };
        },
        console,
    });
    vm.runInContext(fs.readFileSync('Client/cadastros.js', 'utf8'), context);
    const select = names => {
        element('foto').files = names.map(name => new File(['x'], name, { type: 'image/jpeg' }));
        element('foto').onchange();
    };
    const names = () => JSON.parse(vm.runInContext('JSON.stringify(fotos.map(f => f.file.name))', context));
    return { el: element, select, names, revoked, requests };
}

const submit = s => s.el('cadastroForm').onsubmit({ preventDefault() {} });

test('photos accumulate one selection at a time, up to five', () => {
    const s = setup();
    s.select(['1.jpg', '2.jpg']);
    s.select(['3.jpg']);
    assert.deepEqual(s.names(), ['1.jpg', '2.jpg', '3.jpg']);
    // O campo fica vazio para a proxima foto da camera.
    assert.equal(s.el('foto').value, '');
    s.select(['4.jpg', '5.jpg', '6.jpg', '7.jpg']);
    assert.deepEqual(s.names(), ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']);
    assert.match(s.el('status').textContent, /2 ficaram de fora/);
    assert.equal(s.el('fileLabel').textContent, '5 de 5');
    assert.equal(s.el('thumbs').children.length, 5);
});

test('tapping a thumbnail removes that photo and frees its preview', () => {
    const s = setup();
    s.select(['1.jpg', '2.jpg', '3.jpg']);
    s.el('thumbs').children[1].onclick();
    assert.deepEqual(s.names(), ['1.jpg', '3.jpg']);
    assert.deepEqual(s.revoked, ['blob:2.jpg']);
    assert.equal(s.el('thumbs').children.length, 2);
});

test('submit sends the name and every photo in the same field', async () => {
    const s = setup();
    s.el('nome').value = ' Aluno ';
    s.select(['1.jpg', '2.jpg', '3.jpg']);
    await submit(s);
    assert.equal(s.requests.length, 1);
    const [url, options] = s.requests[0];
    assert.equal(url, '/cadastro');
    assert.equal(options.body.get('nome'), 'Aluno');
    assert.deepEqual(options.body.getAll('foto').map(file => file.name), ['1.jpg', '2.jpg', '3.jpg']);
});

test('submit without photos asks for at least one and sends nothing', async () => {
    const s = setup();
    s.el('nome').value = 'Aluno';
    await submit(s);
    assert.equal(s.requests.length, 0);
    assert.match(s.el('status').textContent, /pelo menos uma foto/);
});

test('clearing the form forgets every photo', () => {
    const s = setup();
    s.select(['1.jpg', '2.jpg']);
    s.el('clearBtn').onclick();
    assert.deepEqual(s.names(), []);
    assert.deepEqual(s.revoked, ['blob:1.jpg', 'blob:2.jpg']);
    assert.equal(s.el('fileLabel').textContent, '-');
});
