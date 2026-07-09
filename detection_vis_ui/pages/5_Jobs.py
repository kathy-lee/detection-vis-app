import streamlit as st
import requests
import os
import time

backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

st.set_page_config(
    page_title="Jobs",
    page_icon=":gear:",
)

st.title("Training jobs")
st.caption("Live view of training / retraining jobs. Auto-refreshes every few seconds.")

# Resilient fetch: a transient failure should not crash the page.
try:
    data = requests.get(f"http://{backend_service}:8001/jobs", timeout=10).json()
except Exception:
    st.info("Could not reach the backend, retrying…")
    time.sleep(3)
    st.rerun()

running = data.get("running", [])
queued = data.get("queued", [])
recent = data.get("recent", [])


def _label(job):
    return f"**{job['type']}** · `{job['job_id'][:8]}`"


def render_details(job):
    inp = job.get("inputs") or {}
    with st.expander("Details (input data · split · config)", expanded=False):
        files = inp.get("datafiles") or []
        st.markdown("**Input data:** " +
                    (", ".join(f"{f['name']} ({f['parse']})" for f in files) or "—"))
        st.markdown("**Features:** " + (", ".join(inp.get("features") or []) or "—"))

        split = inp.get("split") or {}
        mode = split.get("mode")
        if mode == "random":
            r = split.get("ratios") or []
            if len(r) == 3:
                st.markdown(f"**Split (random):** train {r[0]:.2f} · val {r[1]:.2f} · test {r[2]:.2f}")
            else:
                st.markdown("**Split:** random")
        elif mode == "sequence":
            st.markdown("**Split (by sequence):**")
            st.json(split.get("files") or {})
        elif mode == "original":
            st.markdown("**Split:** dataset default (original)")

        st.markdown("**Model config:**")
        st.json(inp.get("model_config") or {})
        st.markdown("**Train config:**")
        st.json(inp.get("train_config") or {})


st.subheader("Running")
if not running:
    st.caption("No running jobs.")
for job in running:
    c1, c2, c3 = st.columns([3, 4, 2])
    c1.markdown(_label(job))
    prog = job.get("progress")
    if prog:
        c2.progress((prog["epoch"] - 1) / prog["total_epochs"],
                    text=f"Running epoch {prog['epoch']} of {prog['total_epochs']}")
    else:
        c2.caption("preparing first epoch…")
    # Gracefully stop a running job: the backend signals the worker to kill the
    # work-horse, which frees the GPU automatically.
    if c3.button("Stop", key=f"stop_{job['job_id']}"):
        requests.delete(f"http://{backend_service}:8001/jobs/{job['job_id']}", timeout=10)
        st.rerun()
    render_details(job)

st.subheader("Queued")
if not queued:
    st.caption("No queued jobs.")
for job in queued:
    c1, c2, c3 = st.columns([3, 4, 2])
    c1.markdown(_label(job))
    c2.caption("queued")
    if c3.button("Cancel", key=f"cancel_{job['job_id']}"):
        requests.delete(f"http://{backend_service}:8001/jobs/{job['job_id']}", timeout=10)
        st.rerun()
    render_details(job)

if recent:
    with st.expander("Recent finished / failed", expanded=False):
        for job in recent:
            if job["status"] == "finished":
                st.markdown(f"✅ {_label(job)} → **{job.get('model_name')}**")
            elif job["status"] == "failed":
                st.markdown(f"❌ {_label(job)} → {job.get('error')}")
            else:
                st.markdown(f"• {_label(job)} → {job['status']}")

# Auto-refresh so status/progress stay live without a manual reload.
time.sleep(3)
st.rerun()
