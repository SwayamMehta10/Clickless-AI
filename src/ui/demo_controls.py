"""Streamlit UI for session-scoped demo runtime controls."""

from __future__ import annotations

import streamlit as st

from src.runtime.demo import (
    LIVE,
    MOCKED,
    DemoRuntimeConfig,
    SERVICE_LABELS,
    SERVICE_ORDER,
    apply_preset_selection,
    apply_service_mode,
    get_live_readiness,
)

_PRESET_KEY = "demo_controls_preset"
_LAST_PRESET_KEY = "demo_controls_last_preset"
_CONFIG_KEY = "demo_runtime_config"
_ERRORS_KEY = "demo_runtime_errors"
_NOTICES_KEY = "demo_runtime_notices"


def _service_key(service: str) -> str:
    return f"demo_mode_{service}"


def init_demo_runtime_state() -> None:
    if _CONFIG_KEY not in st.session_state:
        st.session_state[_CONFIG_KEY] = DemoRuntimeConfig.all_mocked()
    if _PRESET_KEY not in st.session_state:
        st.session_state[_PRESET_KEY] = "All Mocked"
    if _LAST_PRESET_KEY not in st.session_state:
        st.session_state[_LAST_PRESET_KEY] = st.session_state[_PRESET_KEY]
    if _ERRORS_KEY not in st.session_state:
        st.session_state[_ERRORS_KEY] = []
    if _NOTICES_KEY not in st.session_state:
        st.session_state[_NOTICES_KEY] = []

    config = st.session_state[_CONFIG_KEY]
    for service in SERVICE_ORDER:
        key = _service_key(service)
        if key not in st.session_state:
            st.session_state[key] = config.mode_for(service)


def get_demo_runtime_config() -> DemoRuntimeConfig:
    init_demo_runtime_state()
    return st.session_state[_CONFIG_KEY]


def _set_runtime_config(config: DemoRuntimeConfig) -> None:
    st.session_state[_CONFIG_KEY] = config
    for service in SERVICE_ORDER:
        st.session_state[_service_key(service)] = config.mode_for(service)


def _apply_preset_if_changed() -> None:
    selected = st.session_state[_PRESET_KEY]
    if selected == st.session_state[_LAST_PRESET_KEY]:
        return

    config, errors = apply_preset_selection(get_demo_runtime_config(), selected)
    _set_runtime_config(config)
    st.session_state[_ERRORS_KEY] = errors
    if selected == "All Mocked":
        st.session_state[_NOTICES_KEY] = ["All services are running in mocked demo mode."]
    elif errors:
        st.session_state[_NOTICES_KEY] = ["All available live services were enabled. Unavailable services stayed mocked."]
    else:
        st.session_state[_NOTICES_KEY] = ["All services are running in live mode."]
    st.session_state[_LAST_PRESET_KEY] = selected
    st.rerun()


def _apply_service_changes() -> None:
    config = get_demo_runtime_config()
    for service in SERVICE_ORDER:
        key = _service_key(service)
        desired = st.session_state[key]
        current = config.mode_for(service)
        if desired == current:
            continue

        updated, error = apply_service_mode(config, service, desired)
        if error:
            st.session_state[key] = current
            st.session_state[_ERRORS_KEY] = [error]
            st.session_state[_NOTICES_KEY] = []
        else:
            _set_runtime_config(updated)
            st.session_state[_ERRORS_KEY] = []
            st.session_state[_NOTICES_KEY] = [f"{SERVICE_LABELS[service]} is now running in {desired.lower()} mode."]
        st.rerun()


def _render_messages() -> None:
    for error in st.session_state.get(_ERRORS_KEY, []):
        st.error(error)
    for notice in st.session_state.get(_NOTICES_KEY, []):
        st.info(notice)


def _render_status_line(service: str) -> None:
    config = get_demo_runtime_config()
    current_mode = config.mode_for(service)
    readiness = get_live_readiness(service)
    if current_mode == LIVE:
        status = "Live is active."
    elif readiness.ready:
        status = f"Live ready: {readiness.message}"
    else:
        status = f"Live unavailable: {readiness.message}"
    st.caption(f"{SERVICE_LABELS[service]}: {current_mode}. {status}")


def render_demo_controls() -> None:
    init_demo_runtime_state()

    with st.expander("Demo Controls", expanded=bool(st.session_state.get(_ERRORS_KEY))):
        st.caption("Use mocked services to run the demo without external setup.")
        _render_messages()

        st.select_slider(
            "Preset",
            options=["All Mocked", "All Live"],
            key=_PRESET_KEY,
        )
        _apply_preset_if_changed()

        for service in SERVICE_ORDER:
            st.select_slider(
                SERVICE_LABELS[service],
                options=[MOCKED, LIVE],
                key=_service_key(service),
            )
            _render_status_line(service)

        _apply_service_changes()
