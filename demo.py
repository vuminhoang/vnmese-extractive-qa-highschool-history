import streamlit as st
from bm25 import BM25, bm25_search
import pandas as pd
from bm25_semantic import clean_sem, three_sub_relevant
from reader import  clean_answer, get_answer, answer_from_model, answer_bm25semantic


# ### Tạo list các document

# In[15]:


def main():
    st.title('Demo web Q&A final NLP')

    st.sidebar.markdown("<h1 style='text-align: left; font-size: 24px;'>Chọn trang</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio('', ['BM25', 'Combination'])
    # Thêm phần tải dữ liệu về
    with open("Text/context.txt", "rb") as file_context:
        st.sidebar.markdown('## **Tải dữ liệu**')
        download_button_context = st.sidebar.download_button(
            label="Tải file context.txt",
            data=file_context,
            file_name="context.txt",
            mime="text/plain"
        )
    with open("Text/test_question.txt", "rb") as file_test:
        download_button_quest = st.sidebar.download_button(
            label="Tải file test_question.txt",
            data=file_test,
            file_name="Text/test_question.txt",
            mime="text/plain"
        )
    if download_button_context or download_button_quest:
        st.sidebar.markdown('File dữ liệu đã được tải xuống.')

    # thanh giá trị
    value = st.sidebar.slider('Chọn số context hiển thị từ 1 đến 5', min_value=1, max_value=5, value=1, step=1)

    if page == 'BM25':
        st.header('Q&A với BM25:')
        ans_input = st.text_input('Mời bạn nhập câu hỏi', 'Nhập vào đây...')
        submit_button = st.button('Gửi')
        st.markdown("---")  # Đường kẻ ngang phân cách
        st.subheader('Kết quả')
        if submit_button:
            answer, correct_context, five_contexts, docs_score = handle_bm25_question(ans_input)
            st.write(answer.upper())

            # Hiển thị 5 contexts phù hợp dưới dạng bảng
            st.subheader('Các contexts phù hợp:')
            if len(five_contexts) >= value:
                # similarity_scores = [0.8, 0.6, 0.9, 0.5, 0.7]  # fake giá trị
                sorted_data = sorted(zip(five_contexts, docs_score), key=lambda x: x[1],
                                     reverse=True)  # sắp xếp theo phù hợp
                sorted_contexts, sorted_scores = zip(*sorted_data)
                data = []
                for index, (context, similarity) in enumerate(zip(sorted_contexts[:value], sorted_scores[:value]),
                                                              start=1):
                    data.append([index, context, similarity])
                df = pd.DataFrame(data, columns=['STT', 'Context', 'Mức độ phù hợp'])
                st.dataframe(df.set_index('STT'), width=800)
            elif len(five_contexts) > 0:
                # similarity_scores = [0.8, 0.6, 0.9, 0.5, 0.7]
                sorted_data = sorted(zip(five_contexts, docs_score), key=lambda x: x[1], reverse=True)
                sorted_contexts, sorted_scores = zip(*sorted_data)
                data = []
                for index, (context, similarity) in enumerate(zip(sorted_contexts[:5], sorted_scores[:5]), start=1):
                    data.append([index, context, similarity])

                while len(data) < 5:
                    data.append(['', '', ''])

                # Tạo DataFrame từ danh sách dữ liệu
                df = pd.DataFrame(data, columns=['STT', 'Context', 'Mức độ phù hợp'])
                st.dataframe(df.set_index('STT'), width=800)
            else:
                st.write('Không tìm thấy contexts phù hợp.')

    elif page == 'Combination':
        st.header('Q&A với phương thức kết hợp:')
        ans_input = st.text_input('Mời bạn nhập câu hỏi', 'Nhập vào đây...')
        submit_button = st.button('Gửi')
        st.markdown("---")  # Đường kẻ ngang phân cách
        st.subheader('Kết quả')
        if submit_button:

            answer, correct_context, contexts = handle_sematic_question(ans_input)
            st.write('Đáp án là:', answer)
            st.write('Context:', correct_context)


def handle_bm25_question(query, limit=5, k1=1.99, b=0.655):
    try:
        i = -1
        contexts, docs_scores = bm25_search(query, limit=limit, k1=k1, b=b)
        for context in contexts:
            i += 1
            results = get_answer(query, context)

            for result in results:
                answer = result[0]['answer']

                if len(answer) > 0:
                    if answer in context:
                        context = context.replace(answer, answer.upper())
                        contexts[i] = contexts[i].replace(answer, answer.upper())
                    return 'Đáp án là: ' + clean_answer(answer).strip(), context.strip(), contexts, docs_scores
        return 'I dont no :<', '-1', list(contexts), docs_scores
    except RuntimeError:
        return "Gặp lỗi rồi :<", '-1', '-1', '-1'


def handle_sematic_question(question):
    question = clean_sem(question)
    arr, doc_scores = bm25_search(question, limit=2, k1=1.5, b=0.75)
    context1 = arr[0]
    context2 = arr[1]
    sub1 = three_sub_relevant(question, context1)
    sub2 = three_sub_relevant(question, context2)
    answer = ''
    context = ''
    for i in sub1:
        a = answer_from_model(question, i)
        if a != '':
            answer = a
            context = i
            return answer, context, sub1
    for i in sub2:
        b = answer_from_model(question, i)
        if b != '':
            answer = b
            context = i
        return answer, context, sub2

    return answer


if __name__ == '__main__':
    main()

# Function to load BM25 model
@st.cache(allow_output_mutation=True)
def load_bm25_model():
    corpus = []
    bm25_model = BM25()
    bm25_model.fit(corpus)
    return bm25_model

