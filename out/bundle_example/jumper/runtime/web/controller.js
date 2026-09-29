export class WebFsm {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebFsmFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_webfsm_free(ptr, 0);
    }
    /**
     * Every binding, as JSON: `[{name, pad, key, on, with, from, leaves}]`.
     * For a host drawing a keypad or wiring a gamepad. A binding is one
     * device's -- a pad button or a key -- and two with one name share a
     * latch.
     * @returns {string}
     */
    bindings() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webfsm_bindings(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The modes that run a model, so a host can check it has a session for
     * every one before the FSM asks for one mid-episode.
     * Replay a bundle's `reference.json` and report where this host differs.
     *
     * The controller only -- it drives the recorded action back in, so the
     * numbers are the observation, the decode and the clamp, never the model.
     * A caller that wants its inference backend checked runs the recorded
     * observations through it and compares against `act` itself.
     *
     * Destroys this controller's state: it is driven through somebody else's
     * frames. Build a fresh one to check, and a fresh one to run.
     * Returns JSON: `{frames, inferences, observation, target, divergent,
     * modeMismatches}`. JSON rather than a struct because the numbers go
     * straight into a test assertion and a panel, and neither wants a
     * generated wrapper type.
     * @param {string} text
     * @returns {string}
     */
    checkReference(text) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.webfsm_checkReference(this.__wbg_ptr, ptr0, len0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * What this controller reads, as JSON, for a page that wants to say so:
     * `{rawInput, keys: [{name, code}], controls, signals: [...]}`. The keys
     * are every key a mode switch or a controls block reads -- modifiers as
     * their two keys -- with the browser
     * code a page receives it under; `controls` is each mode's own account of
     * its bindings -- what the robot's controller prints, a line a mode --
     * joined into one line, modes that read the pad alike named together, or
     * `null`. A page shows these rather than describing the bundle itself.
     * @returns {string}
     */
    inputs() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webfsm_inputs(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * True only while the policy is producing output, as opposed to holding or
     * ramping toward it.
     * @returns {boolean}
     */
    is_running_policy() {
        const ret = wasm.webfsm_is_running_policy(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {Float32Array}
     */
    kd() {
        const ret = wasm.webfsm_kd(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    kp() {
        const ret = wasm.webfsm_kp(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The host's own stop control: a page's Stop button, or Escape.
     *
     * Lets go of everything every mode's operator holds, exactly as a release
     * button does, and says whether there was an operator to let go.
     * A host control rather than a pad button, because a page has no button of
     * the bundle's to press on somebody's behalf: which one releases is the
     * contract's to say, and a page that pressed `B` would be a page that knew.
     * @param {number} now_us
     * @returns {boolean}
     */
    letGo(now_us) {
        const ret = wasm.webfsm_letGo(this.__wbg_ptr, now_us);
        return ret !== 0;
    }
    /**
     * The active mode. For a panel that would otherwise read "running" through
     * a two-second ramp it is not running through.
     * @returns {string}
     */
    mode() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webfsm_mode(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string[]}
     */
    modes() {
        const ret = wasm.webfsm_modes(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * `config` is the FSM TOML, `contracts` a JSON object of
     * `{mode: layout}`, `robot` the JSON above.
     *
     * `trajectories` is `{mode: text}` for the recordings those contracts name
     * in their `reference` block -- the contents of the `*.trajectory.json`
     * files the bundle carries. Omit it for a bundle of ordinary policies; a
     * mode that needs one and does not get it is refused, because the
     * alternative is building its reference terms from zeros, which is a
     * motionless recording that a standing robot tracks perfectly.
     *
     * It is last and optional so a page written against the four-argument form
     * keeps working. Before it existed, a bundle carrying `jumper.jump` or
     * `jumper.dance` built on the board and in `play` and threw here -- the
     * other two hosts had each solved it, one by reading the file and one by
     * being handed the text, and this is the second of those.
     *
     * Every failure here is a `JsError` with the reason in it. There is no
     * fallback: a browser that cannot build this should say so, not run a robot
     * on defaults nobody chose.
     * @param {string} config
     * @param {string} contracts
     * @param {string} robot
     * @param {number} now_us
     * @param {string | null} [trajectories]
     */
    constructor(config, contracts, robot, now_us, trajectories) {
        const ptr0 = passStringToWasm0(config, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(contracts, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(robot, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        var ptr3 = isLikeNone(trajectories) ? 0 : passStringToWasm0(trajectories, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len3 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_new(ptr0, len0, ptr1, len1, ptr2, len2, now_us, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WebFsmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The observation for the mode `tick` just named. Copied out, because the
     * caller is about to hand it to a model that will keep it.
     * @returns {Float32Array}
     */
    observation() {
        const ret = wasm.webfsm_observation(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * One gamepad frame: turn its sticks into a command **using the policy's
     * own contract**.
     *
     * The same call the robot makes (`DeviceHost::on_pad`), for the same
     * reason: a sign corrected in a task's `controls.yaml` has to reach all
     * three hosts, and it only does if none of them interprets the pad
     * itself. A page that mapped sticks to a command before handing it over
     * was a second implementation of the contract, and the two drifted --
     * right down to which names the axes have.
     *
     * For a host with no pad -- a keyboard, an on-screen stick -- `setCommand`
     * is still the way in, and latches nothing, because there is no frame to
     * find an edge in.
     * Deliberately **not** where the buttons are latched. `tick` observes
     * them, once, and this used to observe them too -- so a `rise` or a `fall`
     * fired here and was gone a line later, when `tick` re-observed the same
     * unchanged pad and found no edge. A `toggle` survived it, because a
     * toggle is remembered rather than re-derived, which is why entering a
     * mode worked and the jump's `go` did not: `jump_phase` sat at 0 for the
     * whole of a motion nobody could start, and no host reported anything.
     * @param {number} now_us
     */
    padFrame(now_us) {
        wasm.webfsm_padFrame(this.__wbg_ptr, now_us);
    }
    /**
     * Every control of the pad, what it does in each mode and which keys
     * press it, as JSON (`guide.rs` has the shape). For a page that draws the
     * pad: the account `inputs().controls` gives is one line of prose, and a
     * page that took the contracts and the TOML apart to do better would be
     * a second reading of the bindings.
     * @returns {string}
     */
    padGuide() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webfsm_padGuide(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Joint position targets, wire order. Read after every tick, whether or not
     * it inferred -- a held tick still publishes, and the output filter is
     * still moving.
     * @returns {Float32Array}
     */
    positions() {
        const ret = wasm.webfsm_positions(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {Float32Array} action
     */
    resume(action) {
        const ptr0 = passArrayF32ToWasm0(action, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_resume(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether a **pad** button is down, for a browser reading the Gamepad
     * API. Same contract as `setKey`, with the robot pad service's names --
     * `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3` -- so translate your
     * button index into one of those and the two hosts agree.
     * One raw stick or trigger, by the name the dictionary gives it.
     *
     * `Lx` `Ly` `Rx` `Ry` `LT` `RT`, exactly as the robot's pad service
     * publishes them and as `controller/vocabulary.json` lists them. Pass the
     * value the Gamepad API reports, unchanged: no sign, no deadzone, no
     * scaling. All three of those are the contract's, and a host that applies
     * its own is a second place they live.
     *
     * Returns whether the name is one this hardware reports, so a caller can
     * tell a typo from a stick nobody is touching.
     * @param {string} axis
     * @param {number} value
     * @returns {boolean}
     */
    setAxis(axis, value) {
        const ptr0 = passStringToWasm0(axis, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_setAxis(this.__wbg_ptr, ptr0, len0, value);
        return ret !== 0;
    }
    /**
     * A key, by the browser's own name for it: `KeyboardEvent.code`.
     *
     * **The call a page should make for every key it sees**, down and up,
     * with `repeat` as the event reports it. This controller decides which of
     * them mean anything -- the dictionary it carries says what `KeyW` is --
     * so a bundle whose keyboard this page has never heard of still works,
     * and a page does not need a new build when the dictionary grows a key.
     *
     * `repeat` is a held key's auto-repeat: the same key, still down, which
     * is how both readers take it. A key's control climbs for as long as the
     * key is held -- full after the block's `full_after_s` -- and centres when
     * it comes up, repeated or not; the mode latch reads levels too.
     *
     * Returns whether anything here binds the key, which is what tells a page
     * whether to `preventDefault` it.
     * @param {string} code
     * @param {boolean} down
     * @param {boolean} repeat
     * @param {number} now_us
     * @returns {boolean}
     */
    setKeyCode(code, down, repeat, now_us) {
        const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_setKeyCode(this.__wbg_ptr, ptr0, len0, down, repeat, now_us);
        return ret !== 0;
    }
    /**
     * Whether a key is **down**, not that it was pressed.
     *
     * `rise` and `fall` are both in the vocabulary, so a host has to report
     * both edges; a one-shot "pressed" call could only ever drive half of it.
     * Key names are the contract's (`keypad_1`), not the browser's
     * (`Numpad1`); translating between them is the host's job, the same as it
     * already is for the command keys. Returns whether any binding names this
     * key, so a caller can tell "that key does nothing" from "the rule did not
     * fire" -- two outcomes that look identical on screen.
     * @param {string} key
     * @param {boolean} down
     * @returns {boolean}
     */
    setKey(key, down) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_setKey(this.__wbg_ptr, ptr0, len0, down);
        return ret !== 0;
    }
    /**
     * One pad button going down or coming up, by the dictionary's name --
     * the four d-pad directions included, which the Gamepad API reports as
     * buttons too. Returns whether anything binds it: a mode switch or a
     * chord's modifier, or a mode's task that keeps the direction
     * (`jumper.five_foot`'s arm).
     * @param {string} button
     * @param {boolean} down
     * @returns {boolean}
     */
    setPad(button, down) {
        const ptr0 = passStringToWasm0(button, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_setPad(this.__wbg_ptr, ptr0, len0, down);
        return ret !== 0;
    }
    /**
     * A named quantity the simulator can supply beyond `set_state`.
     *
     * **A page should pass every one it can compute, every frame**, and this
     * keeps what its contracts read. That is the whole of the design: the
     * controller arrives inside an uploaded bundle, the page cannot know
     * which terms that policy observes, and a page that passed only what
     * today's policies need is a page an uploaded bundle outgrows without
     * anything saying so.
     *
     * Read today:
     *
     * | name | values | frame |
     * |---|---|---|
     * | `base_lin_vel` | 3, m/s | body |
     *
     * A robot has no such number -- it needs a state estimator -- which is
     * why no exported policy observes it; a web-only bundle may. Anything
     * else is stored and ignored, and the return says which: `true` when a
     * term here reads it.
     * @param {string} name
     * @param {Float32Array} values
     * @returns {boolean}
     */
    setSignal(name, values) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_setSignal(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        return ret !== 0;
    }
    /**
     * The operator's command, already in physical units.
     *
     * Scaling from a stick's -1..1 to m/s belongs to whoever reads the stick,
     * and in this simulator that is the page, which already does it from the
     * policy's own `command_ranges`. Doing it again here would be a second
     * place the scaling lives.
     *
     * **Never read by a mode whose contract describes controls**
     * (`takesRawInput`): that mode's contract turns the pad and the keys into
     * its command itself, and a page's own mapping on top would be a second
     * one. Only a mode describing none reads this. So a page can send it every
     * frame whatever it loaded -- which is what lets an uploaded bundle carry a
     * controls scheme this page was never written for -- and it keeps the
     * operator fresh either way.
     * @param {number} lin_vel_x
     * @param {number} lin_vel_y
     * @param {number} yaw_rate
     * @param {number} now_us
     */
    set_command(lin_vel_x, lin_vel_y, yaw_rate, now_us) {
        wasm.webfsm_set_command(this.__wbg_ptr, lin_vel_x, lin_vel_y, yaw_rate, now_us);
    }
    /**
     * One frame of robot state, in **wire order**.
     *
     * `tau` is the applied joint torque. In this simulator it is the solver's
     * `actuator_force`, which is the same provenance training had -- unlike the
     * robot, where nothing measures it and the controller reconstructs it from
     * the PD it commanded. `source.rs` is where that difference is stated;
     * passing it here is what makes this the easy case.
     * @param {Float32Array} q
     * @param {Float32Array} qd
     * @param {Float32Array} tau
     * @param {Float32Array} quat
     * @param {Float32Array} gyro
     * @param {number} now_us
     */
    set_state(q, qd, tau, quat, gyro, now_us) {
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(qd, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(tau, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(quat, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passArrayF32ToWasm0(gyro, wasm.__wbindgen_malloc);
        const len4 = WASM_VECTOR_LEN;
        const ret = wasm.webfsm_set_state(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, now_us);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Every transition since the last call, as JSON:
     * `[{at, from, to, rule, why}]`. Draining, so nothing is printed twice.
     * @returns {string}
     */
    take_log() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webfsm_take_log(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether this bundle turns raw input -- the pad and every key -- into a
     * command itself, in any of its modes. When it does, those modes never
     * read `set_command` and a page only has to hand over what it sees:
     * `setAxis`, `setPad`, `padFrame` and `setKeyCode`. On-screen sticks are a
     * pad too, by the same names.
     * @returns {boolean}
     */
    takesRawInput() {
        const ret = wasm.webfsm_takesRawInput(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * One tick. Returns the mode that needs an inference, or `null` when the
     * tick is finished and `positions()` is ready to publish.
     * @param {number} now_us
     * @returns {string | undefined}
     */
    tick(now_us) {
        const ret = wasm.webfsm_tick(this.__wbg_ptr, now_us);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
}
if (Symbol.dispose) WebFsm.prototype[Symbol.dispose] = WebFsm.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_67e7344beaa85059: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./controller_bg.js": import0,
    };
}

const WebFsmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_webfsm_free(ptr, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('controller.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
