# main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from sentence_transformers import SentenceTransformer
from retrieval import load_data, search, similar_papers

st.set_page_config(
    page_title="ScholarFlow",
    page_icon="S",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { background: #f0f4f8; font-family: 'Inter', -apple-system, sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(29,78,216,0.25);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -1.5px;
        margin-bottom: 10px;
    }
    .hero-sub { color: rgba(255,255,255,0.6) !important; font-size: 0.95rem; }

    /* Search input */
    .stTextInput > div > div > input {
        border-radius: 14px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        background: white !important;
        color: #0f172a !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
        transition: all 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59,130,246,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }

    /* Buttons — default (secondary) */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        border: 2px solid #e2e8f0 !important;
        background: white !important;
        color: #374151 !important;
        transition: all 0.15s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
        background: #eff6ff !important;
        box-shadow: 0 1px 6px rgba(59,130,246,0.15) !important;
    }

    /* Buttons — primary */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.5) !important;
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
    }

    /* Paper card container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white !important;
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03) !important;
        overflow: hidden !important;
        transition: box-shadow 0.2s !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.05) !important;
        border-color: #cbd5e1 !important;
    }

    /* Badges */
    .badge {
        display: inline-flex; align-items: center; gap: 3px;
        padding: 3px 9px; border-radius: 6px;
        font-size: 0.72rem; font-weight: 600; margin-right: 5px;
    }
    .b-lm { background: #fef3c7; color: #92400e !important; border: 1px solid #fde68a; }
    .b-vn { background: #eff6ff; color: #1d4ed8 !important; border: 1px solid #bfdbfe; }
    .b-ct { background: #f0fdf4; color: #166534 !important; border: 1px solid #bbf7d0; }
    .b-yr { background: #faf5ff; color: #7c3aed !important; border: 1px solid #e9d5ff; }

    .paper-num { font-size: 0.7rem; font-weight: 700; color: #94a3b8 !important; letter-spacing: 0.5px; text-transform: uppercase; }
    .paper-title { font-size: 1.05rem; font-weight: 700; color: #0f172a !important; line-height: 1.5; margin: 6px 0 10px; }
    .paper-meta { margin-bottom: 8px; }
    .card-divider { border-top: 1px solid #f1f5f9; margin: 12px 0 10px; }

    /* Result count */
    .result-info {
        display: inline-flex; align-items: center; gap: 8px;
        background: white; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 10px 18px;
        font-size: 0.875rem; font-weight: 500; color: #475569 !important;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-info strong { color: #0f172a !important; }
    .result-count {
        background: #eff6ff; color: #1d4ed8 !important;
        padding: 2px 9px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;
    }

    /* Empty state */
    .empty-wrap { text-align: center; padding: 72px 32px; }
    .empty-title { font-size: 1.2rem; font-weight: 600; color: #475569 !important; margin-bottom: 6px; }
    .empty-sub { font-size: 0.875rem; color: #94a3b8 !important; margin-bottom: 24px; }
    .chip {
        display: inline-block; background: white;
        border: 1px solid #e2e8f0; border-radius: 20px;
        padding: 6px 14px; margin: 4px;
        font-size: 0.82rem; color: #3b82f6 !important;
        font-weight: 500; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Detail hero */
    .detail-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(15,23,42,0.2);
    }
    .detail-title {
        font-size: 1.5rem; font-weight: 700;
        color: white !important; line-height: 1.45; margin-bottom: 14px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: white; border-radius: 12px;
        border: 1px solid #e2e8f0; padding: 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] p { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; font-size: 1.05rem !important; }
    [data-testid="stMetricValue"] > div { font-size: 1.05rem !important; }

    /* Section title */
    .sec-title {
        font-size: 0.95rem; font-weight: 700; color: #0f172a !important;
        margin-bottom: 14px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0;
    }

    /* Detail page text */
    .stMarkdown p { color: #111111 !important; }

    /* Expander */
    details summary { font-weight: 600 !important; color: #1e293b !important; }
    [data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; margin-bottom: 8px !important; }

    hr { border-color: #e8edf3 !important; margin: 20px 0 !important; }

    /* Link button (원문 보기) */
    .link-btn {
        display: inline-flex; align-items: center; gap: 5px;
        background: white; border: 2px solid #e2e8f0; border-radius: 10px;
        padding: 7px 14px; font-size: 0.875rem; font-weight: 600;
        color: #374151 !important; text-decoration: none !important;
        transition: all 0.15s; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        cursor: pointer; white-space: nowrap;
    }
    .link-btn:hover {
        border-color: #3b82f6 !important; color: #3b82f6 !important;
        background: #eff6ff !important;
    }
</style>
""", unsafe_allow_html=True)


def is_landmark(paper):
    return paper.get('is_landmark', False)


def truncate(text, max_chars=220):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '…'


def abbrev_venue(venue):
    if not venue:
        return 'arXiv'
    v = venue.lower()
    checks = [
        # arXiv / preprint
        ('arxiv',                                                    'arXiv'),
        ('biorxiv',                                                  'bioRxiv'),
        ('medrxiv',                                                  'medRxiv'),
        ('openai',                                                   'OpenAI'),
        # NLP
        ('annual meeting of the association for computational',      'ACL'),
        ('empirical methods in natural language processing',         'EMNLP'),
        ('north american chapter of the association',                'NAACL'),
        ('naacl',                                                    'NAACL'),
        ('european chapter of the association for computational',    'EACL'),
        ('international conference on computational linguistics',    'COLING'),
        ('language resources and evaluation',                        'LREC'),
        ('transactions of the association for computational',        'TACL'),
        # ML
        ('neural information processing systems',                    'NeurIPS'),
        ('advances in neural information',                           'NeurIPS'),
        ('international conference on machine learning',             'ICML'),
        ('international conference on learning representations',     'ICLR'),
        ('artificial intelligence and statistics',                   'AISTATS'),
        ('journal of machine learning research',                     'JMLR'),
        ('trans. mach. learn. res',                                  'TMLR'),
        # AI
        ('aaai conference on artificial intelligence',               'AAAI'),
        ('aaai spring',                                              'AAAI'),
        ('international joint conference on artificial intelligence', 'IJCAI'),
        # Vision
        ('computer vision and pattern recognition',                  'CVPR'),
        ('cvpr',                                                     'CVPR'),
        ('european conference on computer vision',                   'ECCV'),
        ('eccv',                                                     'ECCV'),
        ('international conference on computer vision',              'ICCV'),
        ('iccv',                                                     'ICCV'),
        ('medical image computing and computer-assisted',            'MICCAI'),
        # Speech / Audio
        ('acoustics, speech',                                        'ICASSP'),
        ('interspeech',                                              'Interspeech'),
        # Robotics
        ('international conference on robotics and automation',      'ICRA'),
        ('robotics: science and systems',                            'RSS'),
        ('conference on robot learning',                             'CoRL'),
        # Data / Web
        ('knowledge discovery and data mining',                      'KDD'),
        ('international acm sigir',                                  'SIGIR'),
        ('sigir',                                                    'SIGIR'),
        ('the web conference',                                       'WWW'),
        ('vldb',                                                     'VLDB'),
        ('sigdial',                                                  'SIGDIAL'),
        # Security
        ('usenix security',                                          'USENIX Sec.'),
        ('ieee symposium on security and privacy',                   'IEEE S&P'),
        ('european symposium on security and privacy',               'Euro S&P'),
        ('computer and communications security',                     'CCS'),
        # IEEE Transactions & Journals
        ('ieee transactions on pattern analysis',                    'TPAMI'),
        ('ieee transactions on neural networks',                     'TNNLS'),
        ('ieee transactions on image processing',                    'TIP'),
        ('ieee transactions on knowledge and data',                  'TKDE'),
        ('ieee/acm transactions on audio',                           'TASLP'),
        ('ieee transactions on',                                     'IEEE Trans.'),
        ('proceedings of the ieee',                                  'Proc. IEEE'),
        ('ieee access',                                              'IEEE Access'),
        # ACM
        ('acm computing surveys',                                    'ACM Surv.'),
        ('communications of the acm',                                'CACM'),
        ('acm multimedia',                                           'ACM MM'),
        ('acm transactions on',                                      'ACM Trans.'),
        # Nature / Science
        ('nature machine intelligence',                              'Nat. Mach. Intell.'),
        ('nature computational',                                     'Nat. Comput. Sci.'),
        ('nature communications',                                    'Nat. Commun.'),
        ('nature medicine',                                          'Nat. Med.'),
        ('nature neuroscience',                                      'Nat. Neurosci.'),
        ('nature',                                                   'Nature'),
        ('science robotics',                                         'Sci. Robot.'),
        ('science',                                                  'Science'),
        # Other ML journals
        ('journal of artificial intelligence research',              'JAIR'),
        ('journal of machine learning',                              'JMLR'),
        ('neural computation',                                       'Neural Comput.'),
        ('neural networks',                                          'Neural Networks'),
        ('neurocomputing',                                           'Neurocomputing'),
        ('international journal of computer vision',                 'IJCV'),
        ('information fusion',                                       'Inf. Fusion'),
        ('artificial intelligence review',                           'AI Rev.'),
        ('pattern recognition',                                      'Pattern Recognit.'),
    ]
    for keyword, abbrev in checks:
        if keyword in v:
            return abbrev
    # 매핑 없으면 원본을 20자 이내로
    return venue[:20]


@st.cache_resource
def load():
    data = load_data()
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return data, model


data, model = load()

if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'query' not in st.session_state:
    st.session_state.query = ''


# ── 논문 상세 페이지 ──
if st.session_state.selected_paper is not None:
    paper = st.session_state.selected_paper
    landmark = is_landmark(paper)

    if st.button("← 검색 결과로 돌아가기"):
        st.session_state.selected_paper = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    lm_badge = '<span class="badge b-lm">원조 논문</span>' if landmark else ''
    venue_badge = f'<span class="badge b-vn">{abbrev_venue(paper["venue"])}</span>'
    st.markdown(f"""
    <div class="detail-hero">
        <div class="detail-title">{paper['title']}</div>
        <div>{lm_badge}{venue_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("인용수", f"{paper['citation_count']:,}회")
    with col2:
        st.metric("발표 연도", paper['published'])
    with col3:
        st.metric("학회/저널", abbrev_venue(paper['venue']))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**저자:** {', '.join(paper['authors'][:5])}")
    st.markdown(f"<a href='{paper['url']}' target='_blank' class='link-btn'>논문 원문 보기 →</a>", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sec-title">초록</div>', unsafe_allow_html=True)
    st.write(paper['abstract'])
    st.divider()

    st.markdown('<div class="sec-title">유사 논문 추천</div>', unsafe_allow_html=True)
    idx = paper.get('idx')
    if idx is not None:
        recs = similar_papers(idx, data, top_k=5)
        for r in recs:
            lm = is_landmark(r)
            prefix = "[원조] " if lm else ""
            with st.expander(f"{prefix}{r['title'][:70]}  ·  {r['citation_count']:,}회 인용"):
                col_m, col_n = st.columns([3, 2])
                with col_m:
                    st.markdown(f"**{abbrev_venue(r['venue'])}** &nbsp;|&nbsp; {', '.join(r['authors'][:3])}")
                    st.markdown(truncate(r['abstract']))
                with col_n:
                    st.markdown(f"**인용수:** {r['citation_count']:,}회")
                    st.markdown(f"**연도:** {r['published']}")
                col_x, col_y = st.columns([2, 8])
                with col_x:
                    if st.button("자세히 보기", key=f"rec_{r['idx']}", type="primary"):
                        st.session_state.selected_paper = {**r, 'idx': r['idx']}
                        st.rerun()
                with col_y:
                    st.markdown(f"<a href='{r['url']}' target='_blank' class='link-btn'>원문 보기 →</a>", unsafe_allow_html=True)
    else:
        st.warning("유사 논문을 불러올 수 없습니다.")


# ── 검색 페이지 ──
else:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">ScholarFlow</div>
        <div class="hero-sub">AI · ML · NLP 논문 검색 및 추천 시스템</div>
    </div>
    """, unsafe_allow_html=True)

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "논문 검색",
            placeholder="논문 주제를 자연어로 입력하세요 (예: BERT language model, transformer attention)",
            label_visibility="collapsed"
        )
    with col_btn:
        search_btn = st.button("검색", use_container_width=True, type="primary")

    if search_btn and query:
        st.session_state.query = query
        with st.spinner("검색 중..."):
            st.session_state.search_results = search(query, data, model, top_k=10)

    if st.session_state.search_results:
        st.markdown(f"""
        <div class="result-info">
            <strong>'{st.session_state.query}'</strong> 검색 결과
            <span class="result-count">{len(st.session_state.search_results)}건</span>
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(st.session_state.search_results):
            landmark = is_landmark(r)
            lm_b = '<span class="badge b-lm">원조 논문</span>' if landmark else ''
            vn_b = f'<span class="badge b-vn">{abbrev_venue(r["venue"])}</span>'
            ct_b = f'<span class="badge b-ct">{r["citation_count"]:,}회 인용</span>'
            yr_b = f'<span class="badge b-yr">{r["published"]}</span>'

            with st.container(border=True):
                st.markdown(f"""
                <div class="paper-num">결과 {i+1}</div>
                <div class="paper-title">{r['title']}</div>
                <div class="paper-meta">{lm_b}{vn_b}{ct_b}{yr_b}</div>
                <div style="font-size:0.82rem; color:#111111; margin-bottom:10px;">{', '.join(r['authors'][:3])}</div>
                <div style="font-size:0.875rem; color:#111111; line-height:1.65; margin-bottom:4px;">{truncate(r['abstract'])}</div>
                <div class="card-divider"></div>
                """, unsafe_allow_html=True)

                col_a, col_b = st.columns([2, 10])
                with col_a:
                    if st.button("자세히 보기", key=f"paper_{i}", type="primary"):
                        st.session_state.selected_paper = {**r, 'idx': r['idx']}
                        st.rerun()
                with col_b:
                    st.markdown(f"<a href='{r['url']}' target='_blank' class='link-btn'>원문 보기 →</a>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-title">논문을 검색해보세요</div>
            <div class="empty-sub">아래 예시 주제를 참고하거나 직접 입력해보세요</div>
            <div>
                <span class="chip">BERT language model</span>
                <span class="chip">transformer attention</span>
                <span class="chip">reinforcement learning</span>
                <span class="chip">retrieval augmented generation</span>
                <span class="chip">GPT text generation</span>
                <span class="chip">contrastive learning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
